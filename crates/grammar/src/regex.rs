//! Regex front-end (full-match semantics).
//!
//! Compiles the practical regex subset used for structured-output
//! constraints (XGrammar, `xgrammar/regex.cpp`, compiles the same
//! subset through its EBNF backend) into the shared Thompson NFA.
//! Matching is **full-match**: a pattern matches a subject iff the
//! entire byte string of the subject is matched (as if wrapped in
//! `^(?:pattern)$`). Because of this, anchors `^` and `$` are accepted
//! and treated as no-ops, and greedy vs lazy quantifiers denote the
//! same language — a trailing `?` after a quantifier (lazy marker) is
//! accepted and ignored.
//!
//! Supported: literals with escapes (`\n \t \r \\ \. \* \+ \? \( \)
//! \[ \] \{ \} \| \^ \$ \- \/` and any other escaped ASCII punctuation,
//! `\d \D \w \W \s \S`, `\xHH`, `\uXXXX` decoded to UTF-8 bytes), `.`
//! (any byte but `\n`), classes `[...]` with ranges, negation and the
//! same escapes, alternation, capturing/non-capturing groups (treated
//! identically), quantifiers `* + ? {m} {m,} {m,n}` (caps as for EBNF:
//! counts are limited to [`crate::GrammarLimits::max_count`] and
//! `{m,}` means `{m, min(m + 2048, max_count)}`).
//!
//! Rejected with a clear `Error::Grammar`: backreferences `\1`..`\9`
//! (and `\0`), word boundaries `\b`/`\B`, lookahead/lookbehind
//! (`(?=`, `(?!`, `(?<=`, `(?<!`), named groups `(?P<...>`, atomic
//! groups `(?>`, inline flags `(?i` and any other `(?` group except
//! `(?:`, possessive quantifier modifiers (`a*+`, `a++`, `a?+`,
//! `{m,n}+` — their match language differs from the greedy ones, so
//! silently treating them as greedy would be unsound), and `\uXXXX`
//! surrogate code points. The empty pattern matches only the empty
//! string.

use crate::ebnf::{lower_ast_no_rules, Ast};
use crate::nfa::{ClassSet, Nfa, NfaBuilder};
use crate::GrammarLimits;
use deepinfer_core::{ensure, Error, Result};

/// A cursor over the pattern bytes (shared by the regex sub-parsers).
struct Cur<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<u8> {
        self.bytes.get(self.pos + off).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }
}

fn hex_val(b: u8) -> Option<u32> {
    (b as char).to_digit(16)
}

fn is_ascii_punct(b: u8) -> bool {
    b.is_ascii_punctuation()
}

fn shorthand_set(c: u8) -> ClassSet {
    let mut s = ClassSet::empty();
    match c {
        b'd' => s.add_range(b'0', b'9'),
        b'w' => {
            s.add_range(b'a', b'z');
            s.add_range(b'A', b'Z');
            s.add_range(b'0', b'9');
            s.add(b'_');
        }
        b's' => {
            s.add_range(0x09, 0x0D);
            s.add(0x20);
        }
        _ => {}
    }
    s
}

/// Parses one escape sequence outside a class. Returns either a byte
/// literal, a shorthand class, or (for `\uXXXX`) a multi-byte UTF-8
/// literal.
enum Escaped {
    /// A single literal byte.
    Byte(u8),
    /// A shorthand class such as `\d`.
    Class(ClassSet),
    /// A multi-byte UTF-8 literal (from `\uXXXX`).
    Utf8(Vec<u8>),
}

fn parse_escape(cur: &mut Cur<'_>) -> Result<Escaped> {
    let start = cur.pos;
    let Some(b'\\') = cur.bump() else {
        return Err(Error::Grammar(format!(
            "expected an escape at offset {start}"
        )));
    };
    let Some(c) = cur.bump() else {
        return Err(Error::Grammar(format!(
            "unterminated escape at offset {start}"
        )));
    };
    match c {
        b'n' => Ok(Escaped::Byte(b'\n')),
        b't' => Ok(Escaped::Byte(b'\t')),
        b'r' => Ok(Escaped::Byte(b'\r')),
        b'd' | b'w' | b's' => Ok(Escaped::Class(shorthand_set(c))),
        b'D' | b'W' | b'S' => {
            let mut s = shorthand_set(c.to_ascii_lowercase());
            s.negate();
            Ok(Escaped::Class(s))
        }
        b'x' => {
            let h1 = cur.bump().and_then(hex_val).ok_or_else(|| {
                Error::Grammar(format!(
                    "\\xHH expects two hex digits at offset {}",
                    cur.pos
                ))
            })?;
            let h2 = cur.bump().and_then(hex_val).ok_or_else(|| {
                Error::Grammar(format!(
                    "\\xHH expects two hex digits at offset {}",
                    cur.pos
                ))
            })?;
            Ok(Escaped::Byte(((h1 << 4) | h2) as u8))
        }
        b'u' => {
            let mut cp: u32 = 0;
            for _ in 0..4 {
                let h = cur.bump().and_then(hex_val).ok_or_else(|| {
                    Error::Grammar(format!(
                        "\\uXXXX expects four hex digits at offset {}",
                        cur.pos
                    ))
                })?;
                cp = (cp << 4) | h;
            }
            ensure!(
                !(0xD800..0xE000).contains(&cp),
                Grammar,
                "\\u{cp:04X} is a surrogate code point; lone surrogate escapes are not valid UTF-8 (offset {start})"
            );
            let ch = char::from_u32(cp).ok_or_else(|| {
                Error::Grammar(format!("invalid \\uXXXX code point at offset {start}"))
            })?;
            let mut buf = [0u8; 4];
            Ok(Escaped::Utf8(ch.encode_utf8(&mut buf).as_bytes().to_vec()))
        }
        c if c.is_ascii_digit() => Err(Error::Grammar(format!(
            "backreference '\\{c}' is not supported (offset {start}); only the regular subset is compiled"
        ))),
        b'b' | b'B' => Err(Error::Grammar(format!(
            "word boundary '\\{c}' is not supported (offset {start})"
        ))),
        c if is_ascii_punct(c) => Ok(Escaped::Byte(c)),
        other => Err(Error::Grammar(format!(
            "unsupported escape '\\{}' at offset {}",
            other as char, start
        ))),
    }
}

/// Parses a class endpoint: a raw byte or an escape. Shorthand escapes
/// inside classes are returned as `Class` for union/negation handling.
enum ClassItem {
    /// A single byte (from a raw byte, a simple escape or `\xHH`).
    Byte(u8),
    /// A shorthand class (`\d` etc.).
    Class(ClassSet),
    /// Multi-byte endpoint (illegal in classes; caller reports).
    #[allow(dead_code)]
    Multi(Vec<u8>),
}

fn parse_class_item(cur: &mut Cur<'_>) -> Result<ClassItem> {
    if cur.peek() == Some(b'\\') {
        match parse_escape(cur)? {
            Escaped::Byte(b) => Ok(ClassItem::Byte(b)),
            Escaped::Class(s) => Ok(ClassItem::Class(s)),
            Escaped::Utf8(v) => Ok(ClassItem::Multi(v)),
        }
    } else if let Some(b) = cur.bump() {
        Ok(ClassItem::Byte(b))
    } else {
        Err(Error::Grammar(format!(
            "unterminated character class at offset {}",
            cur.pos
        )))
    }
}

/// Parses `[...]` / `[^...]`. A `]` as the first item is a literal `]`;
/// a `-` at the start or end of the item list is a literal `-`.
fn parse_class(cur: &mut Cur<'_>) -> Result<Ast> {
    let start = cur.pos;
    cur.bump(); // '['
    let negated = if cur.peek() == Some(b'^') {
        cur.pos += 1;
        true
    } else {
        false
    };
    let mut set = ClassSet::empty();
    let mut first = true;
    loop {
        match cur.peek() {
            None => {
                return Err(Error::Grammar(format!(
                    "unterminated character class at offset {start}"
                )))
            }
            Some(b']') if !first => {
                cur.pos += 1;
                break;
            }
            _ => first = false,
        }
        let item_start = cur.pos;
        let lo = parse_class_item(cur)?;
        // Range: '-' directly before ']' (or EOF) is a literal '-'.
        if cur.peek() == Some(b'-') && cur.peek_at(1) != Some(b']') && cur.peek_at(1).is_some() {
            cur.pos += 1;
            let hi = parse_class_item(cur)?;
            match (lo, hi) {
                (ClassItem::Byte(lo), ClassItem::Byte(hi)) => {
                    ensure!(
                        lo <= hi,
                        Grammar,
                        "invalid class range '{}-{}' (offset {item_start})",
                        lo as char,
                        hi as char
                    );
                    set.add_range(lo, hi);
                }
                _ => {
                    return Err(Error::Grammar(format!(
                        "character class ranges must be single bytes (offset {item_start})"
                    )))
                }
            }
        } else {
            match lo {
                ClassItem::Byte(b) => set.add(b),
                ClassItem::Class(s) => {
                    for b in s.iter() {
                        set.add(b);
                    }
                }
                ClassItem::Multi(_) => {
                    return Err(Error::Grammar(format!(
                        "multi-byte escapes are not allowed inside character classes (offset {item_start})"
                    )))
                }
            }
        }
    }
    if set.is_empty() {
        return Err(Error::Grammar(format!(
            "empty character class at offset {start}"
        )));
    }
    if negated {
        set.negate();
    }
    Ok(Ast::Class(set))
}

/// Parses `{m}`, `{m,}` or `{m,n}` (cursor on `{`).
fn parse_counted(cur: &mut Cur<'_>, max_count: u32) -> Result<(u32, Option<u32>)> {
    // NOTE: the '{' has already been consumed by the postfix-quantifier
    // handler; `cur` points at the first digit.
    let start = cur.pos;
    let m = parse_count_digits(cur, start, max_count)?;
    if cur.peek() == Some(b'}') {
        cur.pos += 1;
        return Ok((m, Some(m)));
    }
    ensure!(
        cur.peek() == Some(b','),
        Grammar,
        "invalid repetition '{{...}}' at offset {start}: expected ',' or '}}'"
    );
    cur.pos += 1;
    if cur.peek() == Some(b'}') {
        cur.pos += 1;
        // {m,} — documented bounded approximation.
        return Ok((m, Some(m.saturating_add(2048).min(max_count))));
    }
    let n = parse_count_digits(cur, start, max_count)?;
    ensure!(
        cur.bump() == Some(b'}'),
        Grammar,
        "invalid repetition '{{...}}' at offset {start}: expected '}}'"
    );
    ensure!(m <= n, Grammar, "repetition has m > n (offset {start})");
    Ok((m, Some(n)))
}

fn parse_count_digits(cur: &mut Cur<'_>, start: usize, max_count: u32) -> Result<u32> {
    let mut v: u64 = 0;
    let mut ndigits = 0;
    while let Some(c) = cur.peek() {
        if !c.is_ascii_digit() {
            break;
        }
        ndigits += 1;
        v = v * 10 + (c - b'0') as u64;
        ensure!(
            v <= max_count as u64,
            CapacityExceeded,
            "repetition bound {v} exceeds the limit of {max_count} (offset {start}); this is a hard cap to keep compilation bounded"
        );
        cur.pos += 1;
    }
    ensure!(
        ndigits > 0,
        Grammar,
        "expected a count in repetition at offset {}",
        cur.pos
    );
    Ok(v as u32)
}

fn is_quant_byte(b: u8) -> bool {
    matches!(b, b'?' | b'*' | b'+' | b'{')
}

/// Parses an alternation. `depth` bounds group nesting.
fn parse_alt(cur: &mut Cur<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let mut branches = vec![parse_seq(cur, max_count, depth, max_depth)?];
    while cur.peek() == Some(b'|') {
        cur.pos += 1;
        branches.push(parse_seq(cur, max_count, depth, max_depth)?);
    }
    if branches.len() == 1 {
        Ok(branches.pop().unwrap_or(Ast::Empty))
    } else {
        Ok(Ast::Alt(branches))
    }
}

/// Parses a (possibly empty) sequence; an empty sequence matches the
/// empty string, so `a|` and `(|b)` are legal.
fn parse_seq(cur: &mut Cur<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let mut items = Vec::new();
    while let Some(c) = cur.peek() {
        if c == b'|' || c == b')' {
            break;
        }
        items.push(parse_item(cur, max_count, depth, max_depth)?);
    }
    match items.len() {
        0 => Ok(Ast::Empty),
        1 => Ok(items.pop().unwrap_or(Ast::Empty)),
        _ => Ok(Ast::Seq(items)),
    }
}

fn parse_item(cur: &mut Cur<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let atom = parse_atom(cur, max_count, depth, max_depth)?;
    if cur.peek().map(is_quant_byte).unwrap_or(false) {
        let q_start = cur.pos;
        let quant = cur.bump().unwrap_or(b'?');
        // Lazy modifier: same language under full-match semantics.
        if cur.peek() == Some(b'?') {
            cur.pos += 1;
        }
        // Possessive modifier: rejected (its match language differs).
        ensure!(
            cur.peek() != Some(b'+'),
            Grammar,
            "possessive quantifier '{}+' is not supported (offset {q_start}); it has different match semantics from the greedy form",
            quant as char
        );
        ensure!(
            !cur.peek().map(is_quant_byte).unwrap_or(false),
            Grammar,
            "unexpected second quantifier after '{}' at offset {}",
            quant as char,
            cur.pos
        );
        let ast = match quant {
            b'?' => Ast::Repeat {
                min: 0,
                max: Some(1),
                inner: Box::new(atom),
            },
            b'*' => Ast::Repeat {
                min: 0,
                max: None,
                inner: Box::new(atom),
            },
            b'+' => Ast::Repeat {
                min: 1,
                max: None,
                inner: Box::new(atom),
            },
            b'{' => {
                let (m, n) = parse_counted(cur, max_count)?;
                // Lazy marker may also follow a counted repetition.
                if cur.peek() == Some(b'?') {
                    cur.pos += 1;
                }
                ensure!(
                    cur.peek() != Some(b'+'),
                    Grammar,
                    "possessive quantifier is not supported (offset {q_start})"
                );
                Ast::Repeat {
                    min: m,
                    max: n,
                    inner: Box::new(atom),
                }
            }
            other => {
                return Err(Error::Grammar(format!(
                    "unexpected quantifier '{}' at offset {}",
                    other as char,
                    cur.pos
                )))
            }
        };
        Ok(ast)
    } else {
        Ok(atom)
    }
}

fn parse_atom(cur: &mut Cur<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let start = cur.pos;
    let Some(c) = cur.peek() else {
        return Err(Error::Grammar(format!(
            "unexpected end of pattern at offset {start}"
        )));
    };
    match c {
        b'.' => {
            cur.pos += 1;
            let mut set = ClassSet::full();
            set.remove(b'\n');
            Ok(Ast::Class(set))
        }
        b'[' => parse_class(cur),
        b'(' => {
            ensure!(
                depth < max_depth,
                CapacityExceeded,
                "group nesting exceeds the recursion bound of {max_depth} (group opened at offset {start})"
            );
            cur.pos += 1;
            if cur.peek() == Some(b'?') {
                match cur.peek_at(1) {
                    Some(b':') => {
                        cur.pos += 2; // non-capturing group
                    }
                    Some(b'=') | Some(b'!') => {
                        return Err(Error::Grammar(format!(
                            "lookahead '(?{}' is not supported (offset {start})",
                            cur.peek_at(1).unwrap_or(b'?') as char
                        )))
                    }
                    Some(b'<') => {
                        return Err(Error::Grammar(format!(
                            "lookbehind '(?<' is not supported (offset {start})"
                        )))
                    }
                    Some(b'P') => {
                        return Err(Error::Grammar(format!(
                            "named group '(?P<' is not supported (offset {start}); use plain groups"
                        )))
                    }
                    Some(b'>') => {
                        return Err(Error::Grammar(format!(
                            "atomic group '(?>' is not supported (offset {start})"
                        )))
                    }
                    other => {
                        return Err(Error::Grammar(format!(
                            "unsupported group extension '(?{}' at offset {start}",
                            other.map(|b| b as char).unwrap_or('?')
                        )))
                    }
                }
            }
            let inner = parse_alt(cur, max_count, depth + 1, max_depth)?;
            ensure!(
                cur.bump() == Some(b')'),
                Grammar,
                "expected ')' to close group opened at offset {start}"
            );
            Ok(inner)
        }
        b')' => Err(Error::Grammar(format!(
            "unmatched ')' at offset {start}"
        )),
        ),
        b'^' | b'$' => {
            // Anchors are no-ops under full-match semantics (documented).
            cur.pos += 1;
            Ok(Ast::Empty)
        }
        b'\\' => match parse_escape(cur)? {
            Escaped::Byte(b) => Ok(Ast::Literal(vec![b])),
            Escaped::Class(s) => Ok(Ast::Class(s)),
            Escaped::Utf8(v) => Ok(Ast::Literal(v)),
        },
        b'{' => {
            // A '{' at atom position is a literal (it cannot be a
            // quantifier because there is nothing to repeat).
            cur.pos += 1;
            Ok(Ast::Literal(vec![b'{']))
        }
        b'*' | b'+' | b'?' => Err(Error::Grammar(format!(
            "quantifier '{}' with nothing to repeat at offset {start}",
            c as char
        ))),
        c => {
            cur.pos += 1;
            Ok(Ast::Literal(vec![c]))
        }
    }
}

/// Compiles a regex pattern to an NFA with full-match semantics.
pub(crate) fn compile_regex_nfa(pattern: &str, limits: &GrammarLimits) -> Result<Nfa> {
    let max_nest = limits.max_recursion_depth as usize;
    let mut cur = Cur {
        bytes: pattern.as_bytes(),
        pos: 0,
    };
    let ast = parse_alt(&mut cur, limits.max_count, 0, max_nest)?;
    ensure!(
        cur.pos == cur.bytes.len(),
        Grammar,
        "unexpected trailing bytes at offset {}",
        cur.pos
    );
    let mut builder = NfaBuilder::new(limits.max_nfa_nodes);
    let frag = lower_ast_no_rules(&mut builder, &ast, limits)?;
    builder.finish(frag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::rng::Philox;

    fn full_match(pattern: &str, subject: &[u8]) -> bool {
        let nfa = compile_regex_nfa(pattern, &GrammarLimits::default()).unwrap();
        // Walk the NFA with an epsilon-closure simulation.
        fn close(nfa: &Nfa, set: &mut Vec<u32>) {
            let mut i = 0;
            while i < set.len() {
                for &e in &nfa.nodes[set[i] as usize].eps {
                    if !set.contains(&e) {
                        set.push(e);
                    }
                }
                i += 1;
            }
        }
        let mut cur = vec![nfa.start];
        close(&nfa, &mut cur);
        for &b in subject {
            let mut next = Vec::new();
            for &n in &cur {
                for &(eb, t) in &nfa.nodes[n as usize].edges {
                    if eb == b && !next.contains(&t) {
                        next.push(t);
                    }
                }
            }
            close(&nfa, &mut next);
            cur = next;
        }
        cur.iter().any(|&n| nfa.nodes[n as usize].accept)
    }

    fn m(pattern: &str, subject: &str) -> bool {
        full_match(pattern, subject.as_bytes())
    }

    #[test]
    fn basics() {
        assert!(m("", ""));
        assert!(!m("", "a"));
        assert!(m("abc", "abc"));
        assert!(!m("abc", "ab"));
        assert!(!m("abc", "abcd"));
        assert!(m("a*", ""));
        assert!(m("a*", "aaaa"));
        assert!(!m("a*", "aaab"));
        assert!(m("a+b", "ab"));
        assert!(m("a+b", "aaab"));
        assert!(!m("a+b", "b"));
        assert!(m("a?b", "b"));
        assert!(m("a?b", "ab"));
        assert!(!m("a?b", "aab"));
    }

    #[test]
    fn alternation_groups_empty_branches() {
        assert!(m("a|b", "a"));
        assert!(m("a|b", "b"));
        assert!(!m("a|b", "c"));
        assert!(m("a|", "a"));
        assert!(m("a|", ""));
        assert!(m("(|b)", ""));
        assert!(m("(ab)*", ""));
        assert!(m("(ab)*", "abab"));
        assert!(!m("(ab)*", "aba"));
        assert!(m("(?:a|b)+", "abba"));
        assert!(m("()", ""));
    }

    #[test]
    fn classes_and_dot() {
        assert!(m("[a-c]+", "abcba"));
        assert!(!m("[a-c]+", "d"));
        assert!(m("[^a-c]", "d"));
        assert!(!m("[^a-c]", "a"));
        assert!(m("[]x]+", "]x"));
        assert!(m("[-x]+", "-"));
        assert!(m("[\\d]+", "012"));
        assert!(!m("[\\d]+", "a"));
        assert!(m("[\\D]", "a"));
        assert!(!m("[\\D]", "5"));
        assert!(m("[\\w\\-]+", "a-b_c"));
        assert!(m("[a-cx-z]+", "azcx"));
        assert!(!m("[a-cx-z]+", "d"));
        assert!(m("a.b", "a_b"));
        assert!(!m("a.b", "a\nb"));
    }

    #[test]
    fn escapes() {
        assert!(m("a\\nb", "a\nb"));
        assert!(m("a\\tb", "a\tb"));
        assert!(m("a\\rb", "a\rb"));
        assert!(m("a\\\\b", "a\\b"));
        assert!(m("\\.\\*\\+\\?\\(\\)\\[\\]\\{\\}\\|\\^\\$\\-\\/", ".*+?()[]{}|^$-/"));
        assert!(m("\\x41\\x42", "AB"));
        assert!(m("\\u0041", "A"));
        assert!(m("\\u00e9", "é")); // 2-byte UTF-8
        assert!(m("\\u20ac", "€")); // 3-byte UTF-8
        // The braced form is EBNF-only; in regex `\u{41}` must be a
        // compile error (stray `{` after the escape is a quantifier
        // position with nothing to repeat... actually `u` then `{41}`).
        assert!(compile_regex_nfa("\\u{41}", &GrammarLimits::default()).is_err());
        assert!(m("a\\wb", "a_b"));
        assert!(m("a\\sb", "a b"));
        assert!(m("a\\db", "a5b"));
        assert!(m("a\\Wb", "a!b"));
        assert!(m("a\\Sb", "axb"));
        assert!(m("a\\Db", "axb"));
    }

    #[test]
    fn counted_repetition() {
        assert!(m("a{3}", "aaa"));
        assert!(!m("a{3}", "aa"));
        assert!(!m("a{3}", "aaaa"));
        assert!(m("[a-c]{2,4}", "abc"));
        assert!(m("[a-c]{2,4}", "abca"));
        assert!(!m("[a-c]{2,4}", "a"));
        assert!(!m("[a-c]{2,4}", "abcab"));
        assert!(m("a{2,}", &"a".repeat(2050)));
        assert!(!m("a{2,}", &"a".repeat(2051)));
        assert!(m("(ab){2}", "abab"));
        // Lazy markers are accepted (same language).
        assert!(m("a*?", ""));
        assert!(m("a+?", "aaa"));
        assert!(m("a{2,3}?", "aa"));
    }

    #[test]
    fn anchors_are_noops() {
        assert!(m("^abc$", "abc"));
        assert!(m("^abc", "abc"));
        assert!(m("abc$", "abc"));
        assert!(m("^", ""));
        assert!(m("$", ""));
    }

    #[test]
    fn rejected_constructs() {
        let cases = [
            "(a)\\1",
            "a\\2",
            "\\bword\\b",
            "\\Bx",
            "(?=a)",
            "(?!a)",
            "(?<=a)",
            "(?<!a)b",
            "(?P<name>a)",
            "(?P=name)",
            "(?>a+)",
            "(?i)abc",
            "(?x)a",
            "a*+",
            "a++",
            "a?+",
            "a{2,3}+",
            "a**",
            "*a",
            "+a",
            "?a",
            "(a",
            "a)",
            "[a",
            "[z-a]",
            "a{2,1}",
            "a{1000000}",
            "a{,2}",
            "\\ud800",
            "\\q",
            "\\f",
            "\\0",
            "a{5000}",
        ];
        for pat in cases {
            let err = compile_regex_nfa(pat, &GrammarLimits::default());
            assert!(err.is_err(), "expected error for pattern {pat:?}");
        }
        // Check a few error messages for clarity.
        let e = compile_regex_nfa("(a)\\1", &GrammarLimits::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("backreference"), "{e}");
        let e = compile_regex_nfa("(?=a)", &GrammarLimits::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("lookahead"), "{e}");
        let e = compile_regex_nfa("(?P<n>x)", &GrammarLimits::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("named group"), "{e}");
        let e = compile_regex_nfa("a*+", &GrammarLimits::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("possessive"), "{e}");
        let e = compile_regex_nfa("a{1000000}", &GrammarLimits::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("limit"), "{e}");
    }

    /// A naive backtracking matcher over the same subset, written
    /// independently of the NFA pipeline for cross-validation.
    mod naive {
        /// Naive AST (deliberately distinct from the compiler's).
        #[derive(Debug, Clone)]
        pub enum Na {
            Empty,
            Lit(Vec<u8>),
            Class(Vec<(u8, u8)>, bool), // ranges, negated
            Seq(Vec<Na>),
            Alt(Vec<Na>),
            Rep(Box<Na>, u32, Option<u32>),
        }

        pub fn parse(pat: &str) -> Na {
            let b = pat.as_bytes();
            let mut pos = 0usize;
            let ast = alt(b, &mut pos);
            ast
        }

        fn alt(b: &[u8], pos: &mut usize) -> Na {
            let mut branches = vec![seq(b, pos)];
            while *pos < b.len() && b[*pos] == b'|' {
                *pos += 1;
                branches.push(seq(b, pos));
            }
            if branches.len() == 1 {
                branches.pop().unwrap()
            } else {
                Na::Alt(branches)
            }
        }

        fn seq(b: &[u8], pos: &mut usize) -> Na {
            let mut items = Vec::new();
            while *pos < b.len() && b[*pos] != b'|' && b[*pos] != b')' {
                items.push(item(b, pos));
            }
            match items.len() {
                0 => Na::Empty,
                1 => items.pop().unwrap(),
                _ => Na::Seq(items),
            }
        }

        fn item(b: &[u8], pos: &mut usize) -> Na {
            let atom = atom(b, pos);
            if *pos >= b.len() {
                return atom;
            }
            match b[*pos] {
                b'*' => {
                    *pos += 1;
                    skip_lazy(b, pos);
                    Na::Rep(Box::new(atom), 0, None)
                }
                b'+' => {
                    *pos += 1;
                    skip_lazy(b, pos);
                    Na::Rep(Box::new(atom), 1, None)
                }
                b'?' => {
                    *pos += 1;
                    skip_lazy(b, pos);
                    Na::Rep(Box::new(atom), 0, Some(1))
                }
                b'{' => {
                    let save = *pos;
                    *pos += 1;
                    let lo = digits(b, pos);
                    if lo.is_none() {
                        *pos = save;
                        return atom;
                    }
                    let lo = lo.unwrap();
                    let hi = if *pos < b.len() && b[*pos] == b'}' {
                        *pos += 1;
                        Some(lo)
                    } else if *pos < b.len() && b[*pos] == b',' {
                        *pos += 1;
                        if *pos < b.len() && b[*pos] == b'}' {
                            *pos += 1;
                            None
                        } else {
                            let h = digits(b, pos).unwrap();
                            assert_eq!(b[*pos], b'}');
                            *pos += 1;
                            Some(h)
                        }
                    } else {
                        *pos = save;
                        return atom;
                    };
                    skip_lazy(b, pos);
                    Na::Rep(Box::new(atom), lo, hi)
                }
                _ => atom,
            }
        }

        fn skip_lazy(b: &[u8], pos: &mut usize) {
            if *pos < b.len() && b[*pos] == b'?' {
                *pos += 1;
            }
        }

        fn digits(b: &[u8], pos: &mut usize) -> Option<u32> {
            let start = *pos;
            let mut v = 0u32;
            while *pos < b.len() && b[*pos].is_ascii_digit() {
                v = v * 10 + (b[*pos] - b'0') as u32;
                *pos += 1;
            }
            if *pos == start {
                None
            } else {
                Some(v)
            }
        }

        fn atom(b: &[u8], pos: &mut usize) -> Na {
            let c = b[*pos];
            match c {
                b'.' => {
                    *pos += 1;
                    Na::Class(vec![(0, b'\n' - 1), (b'\n' + 1, 255)], false)
                }
                b'[' => {
                    *pos += 1;
                    let mut neg = false;
                    if *pos < b.len() && b[*pos] == b'^' {
                        neg = true;
                        *pos += 1;
                    }
                    let mut ranges = Vec::new();
                    let mut first = true;
                    while *pos < b.len() && !(b[*pos] == b']' && !first) {
                        let lo = class_byte(b, pos);
                        if *pos + 1 < b.len() && b[*pos] == b'-' && b[*pos + 1] != b']' {
                            *pos += 1;
                            let hi = class_byte(b, pos);
                            ranges.push((lo, hi));
                        } else {
                            ranges.push((lo, lo));
                        }
                        first = false;
                    }
                    assert_eq!(b[*pos], b']');
                    *pos += 1;
                    Na::Class(ranges, neg)
                }
                b'(' => {
                    *pos += 1;
                    if *pos + 1 < b.len() && b[*pos] == b'?' && b[*pos + 1] == b':' {
                        *pos += 2;
                    }
                    let inner = alt(b, pos);
                    assert_eq!(b[*pos], b')');
                    *pos += 1;
                    inner
                }
                b'^' | b'$' => {
                    *pos += 1;
                    Na::Empty
                }
                b'\\' => {
                    *pos += 1;
                    let e = b[*pos];
                    *pos += 1;
                    match e {
                        b'n' => Na::Lit(vec![b'\n']),
                        b't' => Na::Lit(vec![b'\t']),
                        b'r' => Na::Lit(vec![b'\r']),
                        b'd' => Na::Class(vec![(b'0', b'9')], false),
                        b'D' => Na::Class(vec![(b'0', b'9')], true),
                        b'w' => Na::Class(
                            vec![(b'a', b'z'), (b'A', b'Z'), (b'0', b'9'), (b'_', b'_')],
                            false,
                        ),
                        b'W' => Na::Class(
                            vec![(b'a', b'z'), (b'A', b'Z'), (b'0', b'9'), (b'_', b'_')],
                            true,
                        ),
                        b's' => Na::Class(vec![(0x09, 0x0D), (b' ', b' ')], false),
                        b'S' => Na::Class(vec![(0x09, 0x0D), (b' ', b' ')], true),
                        other => Na::Lit(vec![other]),
                    }
                }
                other => {
                    *pos += 1;
                    Na::Lit(vec![other])
                }
            }
        }

        fn class_byte(b: &[u8], pos: &mut usize) -> u8 {
            if b[*pos] == b'\\' {
                *pos += 1;
                let e = b[*pos];
                *pos += 1;
                match e {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    other => other,
                }
            } else {
                let c = b[*pos];
                *pos += 1;
                c
            }
        }

        fn in_class(ranges: &[(u8, u8)], neg: bool, byte: u8) -> bool {
            let inside = ranges.iter().any(|&(lo, hi)| byte >= lo && byte <= hi);
            inside != neg
        }

        /// Full-match backtracking matcher. `k` limits the repetition
        /// count to keep the search finite (matches the compiler's
        /// `{m,}` cap of m + 2048 for the property-test sizes used).
        pub fn matches(ast: &Na, subject: &[u8]) -> bool {
            fn m(ast: &Na, s: &[u8], k: u32) -> bool {
                match ast {
                    Na::Empty => s.is_empty(),
                    Na::Lit(bytes) => s == bytes.as_slice(),
                    Na::Class(ranges, neg) => {
                        s.len() == 1 && in_class(ranges, *neg, s[0])
                    }
                    Na::Seq(items) => seq_match(items, 0, s, k),
                    Na::Alt(branches) => branches.iter().any(|b| m(b, s, k)),
                    Na::Rep(inner, lo, hi) => rep_match(inner, *lo, *hi, s, 0, k),
                }
            }
            fn seq_match(items: &[Na], idx: usize, s: &[u8], k: u32) -> bool {
                if idx == items.len() {
                    return s.is_empty();
                }
                // Try every split of s between items[idx] and the rest.
                for split in 0..=s.len() {
                    if m(&items[idx], &s[..split], k) && seq_match(items, idx + 1, &s[split..], k)
                    {
                        return true;
                    }
                }
                false
            }
            fn rep_match(
                inner: &Na,
                lo: u32,
                hi: Option<u32>,
                s: &[u8],
                done: u32,
                k: u32,
            ) -> bool {
                if done >= lo && (hi.is_none() || done <= hi.unwrap()) && s.is_empty() {
                    return true;
                }
                if hi.map(|h| done >= h).unwrap_or(false) {
                    return false;
                }
                if done >= k {
                    return false;
                }
                // A repetition body may legitimately match the empty
                // string (e.g. `(?:.)*` inside a `+`), so split 0 is
                // allowed; the `k` fuel plus the `done` counter keep the
                // recursion finite.
                for split in 0..=s.len() {
                    if m(inner, &s[..split], k.min(k.saturating_sub(1))) {
                        if rep_match(inner, lo, hi, &s[split..], done + 1, k) {
                            return true;
                        }
                    }
                }
                false
            }
            m(ast, subject, 64)
        }
    }

    #[test]
    fn classic_cases_vs_naive() {
        let cases: Vec<(&str, Vec<&str>)> = vec![
            ("a*b", vec!["b", "ab", "aaab", "", "a", "aabx"]),
            ("(a|b)+", vec!["a", "b", "abba", "", "c", "abc"]),
            ("[a-c]{2,4}", vec!["a", "ab", "abcd", "abc", "abcde"]),
            ("a.b", vec!["a_b", "a\nb", "ab", "a__b"]),
            ("(ab|a)*", vec!["", "a", "ab", "aba", "abab", "abb"]),
            ("\\d{3}-\\d{4}", vec!["555-1234", "55-1234", "5555-1234"]),
            ("x[0-9]*y", vec!["xy", "x0y", "x0129y", "xay"]),
            ("(a?)*", vec!["", "a", "aa", "b"]),
        ];
        for (pat, subjects) in cases {
            for s in subjects {
                let got = m(pat, s);
                let want = {
                    let ast = naive::parse(pat);
                    naive::matches(&ast, s.as_bytes())
                };
                assert_eq!(got, want, "pattern {pat:?} subject {s:?}");
            }
        }
    }

    #[test]
    fn property_matches_naive_backtracker() {
        deepinfer_core::prop::prop("regex-vs-naive", 0x6E45_0001, 200, |i, rng| {
            let pat = random_pattern(rng, (i % 4 + 1) as usize);
            let ast = naive::parse(&pat);
            let compiled = compile_regex_nfa(&pat, &GrammarLimits::default())
                .map_err(|e| deepinfer_core::Error::Internal(format!("compile {pat:?}: {e}")))?;
            let _ = compiled;
            // Random subjects over {a,b,c}.
            for _ in 0..6 {
                let len = rng.uniform_usize(0, 8);
                let mut s = String::new();
                for _ in 0..len {
                    s.push(['a', 'b', 'c'][rng.uniform_usize(0, 3)]);
                }
                let got = m(&pat, &s);
                let want = naive::matches(&ast, s.as_bytes());
                if got != want {
                    return Err(deepinfer_core::Error::Internal(format!(
                        "pattern {pat:?} subject {s:?}: compiled={got} naive={want}"
                    )));
                }
            }
            Ok(())
        });
    }

    /// Generates a random pattern over the alphabet {a,b,c} using the
    /// supported constructs, nesting up to `depth` groups.
    fn random_pattern(rng: &mut Philox, depth: usize) -> String {
        let choice = rng.next_u32() % 100;
        if depth == 0 || choice < 30 {
            // Atom.
            match rng.next_u32() % 5 {
                0 => "a".to_string(),
                1 => "b".to_string(),
                2 => "[ab]".to_string(),
                3 => "[^a]".to_string(),
                4 => ".".to_string(),
                _ => unreachable!(),
            }
        } else if choice < 45 {
            // Quantified sub-pattern.
            let inner = random_pattern(rng, depth - 1);
            // The inner pattern may itself end in a quantifier, so wrap
            // it in a group before applying the outer quantifier.
            let inner = format!("(?:{inner})");
            match rng.next_u32() % 4 {
                0 => format!("{inner}*"),
                1 => format!("{inner}+"),
                2 => format!("{inner}?"),
                3 => {
                    let lo = rng.next_u32() % 3;
                    match rng.next_u32() % 3 {
                        0 => format!("{inner}{{{lo}}}"),
                        1 => format!("{inner}{{{lo},}}"),
                        _ => format!("{inner}{{{lo},{}}}", lo + rng.next_u32() % 3),
                    }
                }
                _ => unreachable!(),
            }
        } else if choice < 60 {
            // Alternation.
            let a = random_pattern(rng, depth - 1);
            let b = random_pattern(rng, depth - 1);
            format!("{a}|{b}")
        } else if choice < 75 {
            // Group.
            let inner = random_pattern(rng, depth - 1);
            if rng.next_u32() % 2 == 0 {
                format!("({inner})")
            } else {
                format!("(?:{inner})")
            }
        } else {
            // Sequence.
            let a = random_pattern(rng, depth - 1);
            let b = random_pattern(rng, depth - 1);
            format!("{a}{b}")
        }
    }
}
