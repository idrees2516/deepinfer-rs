//! EBNF front-end (xgrammar-style subset).
//!
//! Implements the EBNF dialect accepted by XGrammar
//! (mlc-ai/xgrammar, `xgrammar/ebnf_script/ebnf.h` + `grammar.cc`):
//! rules `name ::= expr;`, alternation `|`, sequencing, grouping,
//! optional `?`, star `*`, plus `+`, bounded repetition `{m}` /
//! `{m,}` / `{m,n}`, double/single-quoted literals with escapes,
//! character classes with negation, and `.` = any byte but `\n`.
//! Comments are `(* ... *)` (non-nested).
//!
//! Rule references (including recursive ones) are compiled by *lazy
//! unrolling*: each reference re-expands the rule body with a rule
//! expansion budget (`GrammarLimits::max_recursion_depth`, default 256)
//! and the global NFA node budget. Unrolling is the only sound way to
//! embed non-right-linear recursion in a finite automaton: it may
//! reject strings deeper than the budget (a documented, sound
//! under-approximation) but never accepts a string outside the grammar.
//! Recursion that cannot terminate (empty language after cutting) is an
//! `Err`. Total lowering recursion — structural nesting plus reference
//! expansion combined — is capped at `8 * max_recursion_depth` so that
//! untrusted grammars can never overflow the stack.
//!
//! Documented limits: counted repetition bounds must satisfy
//! `m <= n <= GrammarLimits::max_count` (4096); `{m,}` compiles as
//! `{m, min(m + 2048, max_count)}`; the designated root rule is `root`
//! if present, else `start` if present, else the first rule.

use crate::nfa::{ClassSet, Fragment, Nfa, NfaBuilder};
use crate::GrammarLimits;
use deepinfer_core::{ensure, Error, Result};
use std::collections::{HashMap, HashSet};

/// Syntax tree shared by the EBNF and regex front-ends.
#[derive(Debug, Clone)]
pub(crate) enum Ast {
    /// The empty string.
    Empty,
    /// An exact byte sequence.
    Literal(Vec<u8>),
    /// One byte from a set.
    Class(ClassSet),
    /// Concatenation.
    Seq(Vec<Ast>),
    /// Alternation.
    Alt(Vec<Ast>),
    /// Repetition: `min` copies plus up to `max - min` optional copies
    /// (`max == None` means unbounded: a trailing Kleene star).
    Repeat {
        /// Mandatory prefix count.
        min: u32,
        /// Optional upper bound (`None` = star).
        max: Option<u32>,
        /// Repeated sub-expression.
        inner: Box<Ast>,
    },
    /// Reference to a rule by name (EBNF only); the `usize` is the byte
    /// offset of the reference for error reporting.
    Ref(String, usize),
}

/// Cursor over the raw grammar bytes.
struct Lexer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(s: &'a str) -> Lexer<'a> {
        Lexer {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

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

    /// Skips whitespace and `(* ... *)` comments (non-nested).
    fn skip_trivia(&mut self) -> Result<()> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => {
                    self.pos += 1;
                }
                Some(b'(') if self.peek_at(1) == Some(b'*') => {
                    let start = self.pos;
                    self.pos += 2;
                    loop {
                        match (self.peek(), self.peek_at(1)) {
                            (Some(b'*'), Some(b')')) => {
                                self.pos += 2;
                                break;
                            }
                            (Some(_), _) => self.pos += 1,
                            (None, _) => {
                                return Err(Error::Grammar(format!(
                                    "unterminated comment starting at offset {start}"
                                )))
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn is_ident_start(b: u8) -> bool {
        b.is_ascii_alphabetic() || b == b'_'
    }

    fn is_ident_cont(b: u8) -> bool {
        Self::is_ident_start(b) || b.is_ascii_digit() || b == b'-'
    }

    fn ident(&mut self) -> Option<String> {
        if self.peek().map(Self::is_ident_start).unwrap_or(false) {
            let start = self.pos;
            while self.peek().map(Self::is_ident_cont).unwrap_or(false) {
                self.pos += 1;
            }
            // The input is `&str`, so this slice is valid UTF-8.
            Some(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
        } else {
            None
        }
    }
}

/// Parses one escape sequence shared by literals and classes.
/// Returns the decoded bytes (one byte for single-byte escapes, the
/// UTF-8 encoding for `\u{XXXX}`). Multi-byte results are only legal in
/// literals, which the caller enforces.
fn parse_escape(lex: &mut Lexer<'_>) -> Result<Vec<u8>> {
    let start = lex.pos;
    let Some(b'\\') = lex.bump() else {
        return Err(Error::Grammar(format!("expected escape at offset {start}")));
    };
    let Some(c) = lex.bump() else {
        return Err(Error::Grammar(format!(
            "unterminated escape at offset {start}"
        )));
    };
    match c {
        b'n' => Ok(vec![b'\n']),
        b't' => Ok(vec![b'\t']),
        b'r' => Ok(vec![b'\r']),
        b'\\' => Ok(vec![b'\\']),
        b'"' => Ok(vec![b'"']),
        b'\'' => Ok(vec![b'\'']),
        b']' => Ok(vec![b']']),
        b'-' => Ok(vec![b'-']),
        b'u' => {
            // Only the braced form \u{XXXX} is supported (documented).
            if lex.bump() != Some(b'{') {
                return Err(Error::Grammar(format!(
                    "expected '{{' after \\u at offset {} (only \\u{{XXXX}} is supported)",
                    lex.pos - 1
                )));
            }
            let mut cp: u32 = 0;
            let mut n = 0;
            while n < 6 {
                match lex.peek() {
                    Some(b'}') => break,
                    Some(h) if h.is_ascii_hexdigit() => {
                        cp = cp * 16 + (h as char).to_digit(16).unwrap_or(0);
                        lex.pos += 1;
                        n += 1;
                    }
                    other => {
                        return Err(Error::Grammar(format!(
                            "invalid \\u{{XXXX}} escape at offset {} (saw {:?})",
                            lex.pos, other
                        )))
                    }
                }
            }
            if n == 0 || lex.bump() != Some(b'}') {
                return Err(Error::Grammar(format!(
                    "invalid \\u{{XXXX}} escape at offset {start}"
                )));
            }
            let ch = char::from_u32(cp).ok_or_else(|| {
                Error::Grammar(format!(
                    "invalid unicode codepoint U+{cp:X} at offset {start}"
                ))
            })?;
            let mut buf = [0u8; 4];
            Ok(ch.encode_utf8(&mut buf).as_bytes().to_vec())
        }
        other => Err(Error::Grammar(format!(
            "unsupported escape '\\{}' at offset {}",
            other as char,
            lex.pos - 1
        ))),
    }
}

/// Parses a quoted literal (`"` or `'`).
fn parse_literal(lex: &mut Lexer<'_>) -> Result<Ast> {
    let start = lex.pos;
    let Some(quote) = lex.bump() else {
        return Err(Error::Grammar(format!(
            "expected a literal at offset {start}"
        )));
    };
    let mut bytes = Vec::new();
    loop {
        match lex.peek() {
            None => {
                return Err(Error::Grammar(format!(
                    "unterminated literal starting at offset {start}"
                )))
            }
            Some(b) if b == quote => {
                lex.pos += 1;
                return Ok(Ast::Literal(bytes));
            }
            Some(b'\\') => {
                let decoded = parse_escape(lex)?;
                bytes.extend_from_slice(&decoded);
            }
            Some(b) => {
                lex.pos += 1;
                bytes.push(b);
            }
        }
    }
}

/// Parses one class item endpoint: a raw byte or an escape. Returns the
/// decoded bytes; multi-byte endpoints are only valid as single items,
/// which the caller checks.
fn parse_class_endpoint(lex: &mut Lexer<'_>, class_start: usize) -> Result<Vec<u8>> {
    if lex.peek() == Some(b'\\') {
        parse_escape(lex)
    } else if let Some(b) = lex.bump() {
        Ok(vec![b])
    } else {
        Err(Error::Grammar(format!(
            "unterminated character class starting at offset {class_start}"
        )))
    }
}

/// Parses a character class `[...]` / `[^...]` (byte-level). A `]` as
/// the first item is a literal `]` (POSIX behaviour); a `-` at the
/// start or end of the item list is a literal `-`.
fn parse_class(lex: &mut Lexer<'_>) -> Result<Ast> {
    let start = lex.pos;
    lex.bump(); // '['
    let negated = if lex.peek() == Some(b'^') {
        lex.pos += 1;
        true
    } else {
        false
    };
    let mut set = ClassSet::empty();
    let mut first = true;
    loop {
        match lex.peek() {
            None => {
                return Err(Error::Grammar(format!(
                    "unterminated character class starting at offset {start}"
                )))
            }
            Some(b']') if !first => {
                lex.pos += 1;
                break;
            }
            _ => first = false,
        }
        // Parse one class item endpoint (raw byte or escape).
        let item_start = lex.pos;
        let lo = parse_class_endpoint(lex, start)?;
        // Range? A '-' directly before ']' (or EOF) is a literal.
        if lex.peek() == Some(b'-') && lex.peek_at(1) != Some(b']') && lex.peek_at(1).is_some() {
            lex.pos += 1; // consume '-'
            let hi = parse_class_endpoint(lex, start)?;
            ensure!(
                lo.len() == 1 && hi.len() == 1,
                Grammar,
                "character class ranges must be single bytes (offset {item_start})"
            );
            let (lo, hi) = (lo[0], hi[0]);
            ensure!(
                lo <= hi,
                Grammar,
                "invalid class range '{}'-'{}' (offset {item_start})",
                lo as char,
                hi as char
            );
            set.add_range(lo, hi);
        } else {
            ensure!(
                lo.len() == 1,
                Grammar,
                "multi-byte characters are not allowed inside character classes (offset {item_start}); use them in literals instead"
            );
            set.add(lo[0]);
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

/// Parses `{m}`, `{m,}` or `{m,n}` (the cursor is on `{`).
/// Returns `(m, Option<n>)`.
fn parse_counted(lex: &mut Lexer<'_>, max_count: u32) -> Result<(u32, Option<u32>)> {
    // NOTE: the '{' has already been consumed by `parse_item`; `lex`
    // points at the first digit (skipping trivia).
    let start = lex.pos;
    lex.skip_trivia()?;
    let m = parse_count_digits(lex, start, max_count)?;
    lex.skip_trivia()?;
    if lex.peek() == Some(b'}') {
        lex.pos += 1;
        return Ok((m, Some(m)));
    }
    ensure!(
        lex.peek() == Some(b','),
        Grammar,
        "expected ',' or '}}' in counted repetition at offset {}",
        lex.pos
    );
    lex.pos += 1;
    lex.skip_trivia()?;
    if lex.peek() == Some(b'}') {
        lex.pos += 1;
        // {m,} — bounded approximation (documented).
        return Ok((m, Some(bounded_open_ended(m, max_count)?)));
    }
    let n = parse_count_digits(lex, start, max_count)?;
    lex.skip_trivia()?;
    ensure!(
        lex.bump() == Some(b'}'),
        Grammar,
        "expected '}}' in counted repetition at offset {}",
        lex.pos - 1
    );
    ensure!(
        m <= n,
        Grammar,
        "counted repetition has m > n (offset {start})"
    );
    Ok((m, Some(n)))
}

fn parse_count_digits(lex: &mut Lexer<'_>, start: usize, max_count: u32) -> Result<u32> {
    let mut v: u64 = 0;
    let mut ndigits = 0;
    while let Some(c) = lex.peek() {
        if !c.is_ascii_digit() {
            break;
        }
        ndigits += 1;
        v = v * 10 + (c - b'0') as u64;
        ensure!(
            v <= max_count as u64,
            CapacityExceeded,
            "counted repetition bound {v} exceeds the limit of {max_count} (offset {start}); this is a hard cap to keep compilation bounded"
        );
        lex.pos += 1;
    }
    ensure!(
        ndigits > 0,
        Grammar,
        "expected a count in counted repetition at offset {}",
        lex.pos
    );
    Ok(v as u32)
}

/// `{m,}` compiles as `{m, min(m + 2048, max_count)}` (documented
/// bounded approximation; errors when `m` alone exceeds the cap).
fn bounded_open_ended(m: u32, max_count: u32) -> Result<u32> {
    Ok(m.saturating_add(2048).min(max_count))
}

fn is_postfix_byte(b: u8) -> bool {
    matches!(b, b'?' | b'*' | b'+' | b'{')
}

/// Parses an alternation. `depth` is the group-nesting depth, bounded
/// to keep the recursive-descent parser total on untrusted input.
fn parse_alt(lex: &mut Lexer<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let mut branches = vec![parse_seq(lex, max_count, depth, max_depth)?];
    loop {
        lex.skip_trivia()?;
        if lex.peek() == Some(b'|') {
            lex.pos += 1;
            branches.push(parse_seq(lex, max_count, depth, max_depth)?);
        } else {
            break;
        }
    }
    if branches.len() == 1 {
        Ok(branches.pop().unwrap_or(Ast::Empty))
    } else {
        Ok(Ast::Alt(branches))
    }
}

/// Parses a sequence of items (possibly empty, ended by `|`, `)`, `;`
/// or EOF; an empty sequence is an error unless it is the empty
/// literal `""`).
fn parse_seq(lex: &mut Lexer<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let start = lex.pos;
    let mut items = Vec::new();
    loop {
        lex.skip_trivia()?;
        match lex.peek() {
            None | Some(b'|') | Some(b')') | Some(b';') => break,
            _ => {}
        }
        items.push(parse_item(lex, max_count, depth, max_depth)?);
    }
    ensure!(
        !items.is_empty(),
        Grammar,
        "expected an expression at offset {start} (use \"\" for the empty string)"
    );
    if items.len() == 1 {
        Ok(items.pop().unwrap_or(Ast::Empty))
    } else {
        Ok(Ast::Seq(items))
    }
}

fn parse_item(lex: &mut Lexer<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let atom = parse_atom(lex, max_count, depth, max_depth)?;
    lex.skip_trivia()?;
    if lex.peek().map(is_postfix_byte).unwrap_or(false) {
        let quant = lex.bump().unwrap_or(b'?');
        // A second quantifier directly after is an error.
        lex.skip_trivia()?;
        ensure!(
            !lex.peek().map(is_postfix_byte).unwrap_or(false),
            Grammar,
            "unexpected second quantifier after '{}' at offset {}",
            quant as char,
            lex.pos
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
                let (m, n) = parse_counted(lex, max_count)?;
                Ast::Repeat {
                    min: m,
                    max: n,
                    inner: Box::new(atom),
                }
            }
            // Unreachable in practice (checked above) but kept as a
            // total, panic-free fallback.
            other => {
                return Err(Error::Grammar(format!(
                    "unexpected quantifier '{}' at offset {}",
                    other as char,
                    lex.pos
                )))
            }
        };
        Ok(ast)
    } else {
        Ok(atom)
    }
}

fn parse_atom(lex: &mut Lexer<'_>, max_count: u32, depth: usize, max_depth: usize) -> Result<Ast> {
    let start = lex.pos;
    let Some(c) = lex.peek() else {
        return Err(Error::Grammar(format!(
            "unexpected end of grammar at offset {start}"
        )));
    };
    match c {
        b'"' | b'\'' => parse_literal(lex),
        b'[' => parse_class(lex),
        b'.' => {
            lex.pos += 1;
            let mut set = ClassSet::full();
            set.remove(b'\n');
            Ok(Ast::Class(set))
        }
        b'(' => {
            ensure!(
                depth < max_depth,
                CapacityExceeded,
                "expression nesting exceeds the recursion bound of {max_depth} (group opened at offset {start})"
            );
            lex.pos += 1;
            let inner = parse_alt(lex, max_count, depth + 1, max_depth)?;
            ensure!(
                lex.bump() == Some(b')'),
                Grammar,
                "expected ')' to close group opened at offset {start}"
            );
            Ok(inner)
        }
        b if Lexer::is_ident_start(b) => {
            let name = lex.ident().unwrap_or_default();
            Ok(Ast::Ref(name, start))
        }
        other => Err(Error::Grammar(format!(
            "unexpected byte '{}' at offset {start}",
            other as char
        ))),
    }
}

/// A parsed grammar: rules in source order plus a name index.
pub(crate) struct EbnfGrammar {
    /// `(name, body)` pairs in source order.
    pub(crate) rules: Vec<(String, Ast)>,
    /// `name -> rule index` for O(1) reference resolution.
    pub(crate) index: HashMap<String, usize>,
    /// Index of the designated root rule.
    pub(crate) root: usize,
}

/// Parses a whole EBNF grammar.
pub(crate) fn parse_ebnf(text: &str, limits: &GrammarLimits) -> Result<EbnfGrammar> {
    let max_count = limits.max_count;
    let max_nest = limits.max_recursion_depth as usize;
    let mut lex = Lexer::new(text);
    let mut rules: Vec<(String, Ast)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    loop {
        lex.skip_trivia()?;
        if lex.peek().is_none() {
            break;
        }
        let name_pos = lex.pos;
        let Some(name) = lex.ident() else {
            return Err(Error::Grammar(format!(
                "expected a rule name at offset {name_pos}"
            )));
        };
        ensure!(
            seen.insert(name.clone()),
            Grammar,
            "duplicate rule '{name}' at offset {name_pos}"
        );
        lex.skip_trivia()?;
        ensure!(
            lex.peek() == Some(b':') && lex.peek_at(1) == Some(b':') && lex.peek_at(2) == Some(b'='),
            Grammar,
            "expected '::=' after rule name '{name}' at offset {}",
            lex.pos
        );
        lex.pos += 3;
        let body = parse_alt(&mut lex, max_count, 0, max_nest)?;
        lex.skip_trivia()?;
        if lex.peek() == Some(b';') {
            lex.pos += 1;
        }
        rules.push((name, body));
    }
    ensure!(
        !rules.is_empty(),
        Grammar,
        "grammar contains no rules (offset 0)"
    );
    let index: HashMap<String, usize> = rules
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.clone(), i))
        .collect();
    // Root selection: "root" > "start" > first rule (documented).
    let root = index
        .get("root")
        .copied()
        .or_else(|| index.get("start").copied())
        .unwrap_or(0);
    Ok(EbnfGrammar { rules, index, root })
}

/// Context threaded through AST lowering: the rule table, its index and
/// the recursion budgets.
pub(crate) struct LowerCtx<'a> {
    /// `(name, body)` pairs in source order.
    pub(crate) rules: &'a [(String, Ast)],
    /// `name -> rule index`.
    pub(crate) index: &'a HashMap<String, usize>,
    /// Rule-expansion budget (per reference chain).
    pub(crate) max_ref_depth: usize,
    /// Combined structural + reference recursion budget (stack guard).
    pub(crate) max_total_depth: usize,
}

impl<'a> LowerCtx<'a> {
    /// Builds a context for a grammar parsed with `limits`.
    pub(crate) fn new(grammar: &'a EbnfGrammar, limits: &GrammarLimits) -> LowerCtx<'a> {
        LowerCtx {
            rules: &grammar.rules,
            index: &grammar.index,
            max_ref_depth: limits.max_recursion_depth as usize,
            max_total_depth: limits.max_recursion_depth as usize * 8,
        }
    }
}

/// Lowers an AST to an NFA fragment, resolving rule references by lazy
/// expansion. `depth` counts every recursive descent (structural and
/// references) and is capped by `ctx.max_total_depth` — a hard,
/// non-negotiable stack guard. `ref_depth` counts reference expansions
/// only; when it exceeds the budget the reference is *cut* (treated as
/// the empty language) and `cut` is set, which the caller turns into an
/// error if the whole grammar ends up empty. `None` fragments denote
/// dead branches (empty language).
pub(crate) fn lower_ast(
    builder: &mut NfaBuilder,
    ast: &Ast,
    ctx: &LowerCtx<'_>,
    depth: usize,
    ref_depth: usize,
    cut: &mut bool,
) -> Result<Option<Fragment>> {
    ensure!(
        depth < ctx.max_total_depth,
        CapacityExceeded,
        "grammar lowering recursion exceeds the total depth bound of {} (deeply nested or deeply chained rules); refusing to risk unbounded recursion",
        ctx.max_total_depth
    );
    match ast {
        Ast::Empty => builder.frag_empty().map(Some),
        Ast::Literal(bytes) => builder.frag_literal(bytes).map(Some),
        Ast::Class(set) => builder.frag_class(set),
        Ast::Seq(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for it in items {
                parts.push(lower_ast(builder, it, ctx, depth + 1, ref_depth, cut)?);
            }
            builder.seq(parts)
        }
        Ast::Alt(branches) => {
            let mut parts = Vec::with_capacity(branches.len());
            for br in branches {
                parts.push(lower_ast(builder, br, ctx, depth + 1, ref_depth, cut)?);
            }
            builder.alt(parts)
        }
        Ast::Repeat { min, max, inner } => {
            lower_repeat(builder, *min, *max, inner, ctx, depth, ref_depth, cut)
        }
        Ast::Ref(name, pos) => {
            let Some(&idx) = ctx.index.get(name) else {
                return Err(Error::Grammar(format!(
                    "unknown rule '{name}' referenced at offset {pos}"
                )));
            };
            if ref_depth >= ctx.max_ref_depth {
                // Sound under-approximation: stop unrolling here. If the
                // language survives via a base case the grammar still
                // compiles (accepting only up to the budget); otherwise
                // the caller reports the non-termination.
                *cut = true;
                return Ok(None);
            }
            lower_ast(
                builder,
                &ctx.rules[idx].1,
                ctx,
                depth + 1,
                ref_depth + 1,
                cut,
            )
        }
    }
}

fn lower_repeat(
    builder: &mut NfaBuilder,
    min: u32,
    max: Option<u32>,
    inner: &Ast,
    ctx: &LowerCtx<'_>,
    depth: usize,
    ref_depth: usize,
    cut: &mut bool,
) -> Result<Option<Fragment>> {
    if min == 0 && max == Some(0) {
        // {0,0}: the empty string.
        return builder.frag_empty().map(Some);
    }
    // Mandatory prefix: min copies.
    let mut parts: Vec<Option<Fragment>> = Vec::new();
    for _ in 0..min {
        match lower_ast(builder, inner, ctx, depth + 1, ref_depth, cut)? {
            Some(f) => parts.push(Some(f)),
            None => return Ok(None), // dead inner, min > 0 => dead
        }
    }
    match max {
        None => {
            // min copies + star (exact); each copy is a fresh expansion
            // so the loop body shares nothing with the prefix.
            match lower_ast(builder, inner, ctx, depth + 1, ref_depth, cut)? {
                Some(f) => {
                    let st = builder.star(f)?;
                    parts.push(Some(st));
                }
                None => {
                    if parts.is_empty() {
                        return builder.frag_empty().map(Some); // dead* = eps
                    }
                }
            }
            builder.seq(parts)
        }
        Some(n) => {
            // min mandatory + (n - min) optional copies, each a fresh
            // expansion of the inner AST.
            for _ in min..n {
                match lower_ast(builder, inner, ctx, depth + 1, ref_depth, cut)? {
                    Some(f) => parts.push(Some(builder.optional(f)?)),
                    None => break, // dead optionals simply drop out
                }
            }
            builder.seq(parts)
        }
    }
}

/// Lowers a reference-free AST (used by the regex front-end) with the
/// budgets implied by `limits`. The empty rule index is never
/// consulted because reference-free ASTs contain no `Ast::Ref`.
pub(crate) fn lower_ast_no_rules(
    builder: &mut NfaBuilder,
    ast: &Ast,
    limits: &GrammarLimits,
) -> Result<Option<Fragment>> {
    static EMPTY_RULES: &[(String, Ast)] = &[];
    let index: HashMap<String, usize> = HashMap::new();
    let ctx = LowerCtx {
        rules: EMPTY_RULES,
        index: &index,
        max_ref_depth: limits.max_recursion_depth as usize,
        max_total_depth: limits.max_recursion_depth as usize * 8,
    };
    let mut cut = false;
    lower_ast(builder, ast, &ctx, 0, 0, &mut cut)
}

/// Compiles EBNF text to an NFA (root rule lowered; acceptance marked).
pub(crate) fn compile_ebnf_nfa(text: &str, limits: &GrammarLimits) -> Result<Nfa> {
    let grammar = parse_ebnf(text, limits)?;
    let ctx = LowerCtx::new(&grammar, limits);
    let mut builder = NfaBuilder::new(limits.max_nfa_nodes);
    let mut cut = false;
    let frag = lower_ast(
        &mut builder,
        &grammar.rules[grammar.root].1,
        &ctx,
        0,
        0,
        &mut cut,
    )?;
    if frag.is_none() && cut {
        return Err(Error::Grammar(format!(
            "rule '{}' is recursive with no terminating base case within the expansion depth bound of {}",
            grammar.rules[grammar.root].0, limits.max_recursion_depth
        )));
    }
    builder.finish(frag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepted(nfa: &Nfa, s: &[u8]) -> bool {
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
        close(nfa, &mut cur);
        for &b in s {
            let mut next = Vec::new();
            for &n in &cur {
                for &(eb, t) in &nfa.nodes[n as usize].edges {
                    if eb == b && !next.contains(&t) {
                        next.push(t);
                    }
                }
            }
            close(nfa, &mut next);
            cur = next;
        }
        cur.iter().any(|&n| nfa.nodes[n as usize].accept)
    }

    fn compile(text: &str) -> Nfa {
        compile_ebnf_nfa(text, &GrammarLimits::default()).unwrap()
    }

    
#[test]
fn diag_plus_repro() {
    let nfa = compile(r#"root ::= "c"+;"#);
    eprintln!("start={} nodes={}", nfa.start, nfa.nodes.len());
    for (i, n) in nfa.nodes.iter().enumerate() {
        eprintln!("node {i}: accept={} eps={:?} edges={:?}", n.accept, n.eps, n.edges);
    }
    assert!(!accepted(&nfa, b""));
    assert!(accepted(&nfa, b"c"));
}

    #[test]
    fn simple_alternation_and_plus() {
        let nfa = compile(r#"root ::= "ab" | "c"+;"#);
        assert!(accepted(&nfa, b"ab"));
        assert!(accepted(&nfa, b"c"));
        assert!(accepted(&nfa, b"cccc"));
        assert!(!accepted(&nfa, b""));
        assert!(!accepted(&nfa, b"a"));
        assert!(!accepted(&nfa, b"cab"));
    }

    #[test]
    fn json_ish_grammar() {
        let text = r#"
            root ::= object;
            object ::= "{" ws ( member ( ws "," ws member )* )? ws "}";
            member ::= string ws ":" ws value;
            value ::= object | array | string | number | boolean | null;
            array ::= "[" ws ( value ( ws "," ws value )* )? ws "]";
            string ::= "\"" [^"\\\\\u{0}-\u{1F}]* "\"";
            number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?;
            boolean ::= "true" | "false";
            null ::= "null";
            ws ::= [ \t\n\r]*;
        "#;
        // Mutual recursion through two branches is exponential under
        // bounded unrolling; a practical nesting depth keeps it small.
        let lim = GrammarLimits {
            max_recursion_depth: 12,
            ..GrammarLimits::default()
        };
        let nfa = compile_ebnf_nfa(text, &lim).unwrap();
        assert!(accepted(&nfa, br#"{}"#));
        assert!(accepted(&nfa, br#"{"a": [1, true, null], "b": {"c": "d"}}"#));
        assert!(accepted(&nfa, br#"{"n": -12.5}"#));
        assert!(!accepted(&nfa, br#"{"a" 1}"#));
        assert!(!accepted(&nfa, br#"{"a": }"#));
        assert!(!accepted(&nfa, br#"{"a": 1,}"#));
        assert!(!accepted(&nfa, b"[1 2]"));
    }

    #[test]
    fn recursive_parens() {
        let nfa = compile(r#"root ::= "(" root ")" | ""; "#);
        assert!(accepted(&nfa, b""));
        assert!(accepted(&nfa, b"()"));
        assert!(accepted(&nfa, b"(((())))"));
        assert!(!accepted(&nfa, b"(()"));
        assert!(!accepted(&nfa, b"()("));
        // Documented approximation: nesting is unrolled up to the
        // expansion budget (256), so absurdly deep nesting is rejected.
        let deep = "(".repeat(300) + &")".repeat(300);
        assert!(!accepted(&nfa, deep.as_bytes()));
        let ok_deep = "(".repeat(255) + &")".repeat(255);
        assert!(accepted(&nfa, ok_deep.as_bytes()));
    }

    #[test]
    fn right_recursion_with_base() {
        let nfa = compile(r#"s ::= "x" | s "y";"#);
        assert!(accepted(&nfa, b"x"));
        assert!(accepted(&nfa, b"xyy"));
        assert!(!accepted(&nfa, b"y"));
        assert!(!accepted(&nfa, b""));
    }

    #[test]
    fn non_terminating_recursion_errors() {
        assert!(compile_ebnf_nfa(r#"root ::= root;"#, &GrammarLimits::default()).is_err());
        assert!(compile_ebnf_nfa(r#"root ::= "x" root;"#, &GrammarLimits::default()).is_err());
        assert!(compile_ebnf_nfa(
            r#"a ::= b; b ::= a;"#,
            &GrammarLimits::default()
        )
        .is_err());
    }

    #[test]
    fn escapes() {
        let nfa = compile(r#"root ::= "a\nb\t\"c\"\u{41}";"#);
        assert!(accepted(&nfa, b"a\nb\t\"c\"A"));
        assert!(!accepted(&nfa, b"a\nb\t\"c\"a"));
        // Unsupported escapes are a hard error (sound rejection).
        assert!(compile_ebnf_nfa(
            r#"root ::= '\x';"#,
            &GrammarLimits::default()
        )
        .is_err());
    }

    #[test]
    fn classes_negation_ranges() {
        let nfa = compile(r#"root ::= [a-c0-9_]+;"#);
        assert!(accepted(&nfa, b"abc90_"));
        assert!(!accepted(&nfa, b"d"));
        let n2 = compile(r#"root ::= [^abc];"#);
        assert!(accepted(&n2, b"d"));
        assert!(accepted(&n2, &[0xFF]));
        assert!(!accepted(&n2, b"a"));
        let n3 = compile(r#"root ::= .;"#);
        assert!(accepted(&n3, b"a"));
        assert!(!accepted(&n3, b"\n"));
        // ']' as the first class item is a literal.
        let n4 = compile(r#"root ::= []x]+;"#);
        assert!(accepted(&n4, b"]x]"));
        assert!(!accepted(&n4, b"y"));
        // '-' at the boundaries is a literal.
        let n5 = compile(r#"root ::= [-a]+;"#);
        assert!(accepted(&n5, b"-a-"));
        assert!(!accepted(&n5, b"b"));
    }

    #[test]
    fn counted_repetition() {
        let nfa = compile(r#"root ::= "ab"{3};"#);
        assert!(accepted(&nfa, b"ababab"));
        assert!(!accepted(&nfa, b"abab"));
        let n2 = compile(r#"root ::= "x"{2,4};"#);
        assert!(accepted(&n2, b"xx"));
        assert!(accepted(&n2, b"xxxx"));
        assert!(!accepted(&n2, b"x"));
        assert!(!accepted(&n2, b"xxxxx"));
        let n3 = compile(r#"root ::= "y"{0,2};"#);
        assert!(accepted(&n3, b""));
        assert!(accepted(&n3, b"yy"));
        let n5 = compile(r#"root ::= "z"{0,0};"#);
        assert!(accepted(&n5, b""));
        assert!(!accepted(&n5, b"z"));
        // {m,} = {m, min(m+2048, 4096)}: y{2,} accepts 2..=2050 copies.
        let n4 = compile(r#"root ::= "y"{2,};"#);
        assert!(accepted(&n4, "y".repeat(2050).as_bytes()));
        assert!(!accepted(&n4, "y".repeat(2051).as_bytes()));
    }

    #[test]
    fn count_cap_fails_fast() {
        let err = compile_ebnf_nfa(r#"root ::= "a"{1000000};"#, &GrammarLimits::default());
        assert!(matches!(err, Err(Error::CapacityExceeded(_))));
        let err2 = compile_ebnf_nfa(r#"root ::= "a"{5000};"#, &GrammarLimits::default());
        assert!(matches!(err2, Err(Error::CapacityExceeded(_))));
        // Nested counted repetition blows the NFA node budget.
        let nested = r#"root ::= (("a"|"b"){200}){200};"#;
        assert!(matches!(
            compile_ebnf_nfa(nested, &GrammarLimits::default()),
            Err(Error::CapacityExceeded(_))
        ));
    }

    #[test]
    fn comments_and_root_selection() {
        let nfa = compile("(* top *) root ::= \"x\" (* mid *) ; tail ::= \"y\";");
        assert!(accepted(&nfa, b"x"));
        assert!(!accepted(&nfa, b"y"));
        let n2 = compile(r#"first ::= "1"; start ::= "2";"#);
        assert!(accepted(&n2, b"2"));
        let n3 = compile(r#"first ::= "1"; root ::= "3"; start ::= "2";"#);
        assert!(accepted(&n3, b"3"));
    }

    #[test]
    fn parse_errors_carry_positions() {
        let cases = [
            (r#"root ::= "a" "#.to_string() + r#"; other ::= ;"#, "expected an expression"),
            (r#"root ::= unknown;"#.to_string(), "unknown rule"),
            (r#"root ::= "a"#.to_string(), "unterminated literal"),
            (r#"root ::= "a\q";"#.to_string(), "unsupported escape"),
            (r#"root ::= ( "a" ;"#.to_string(), "expected ')'"),
            (r#"root ::= "a"{4,2};"#.to_string(), "m > n"),
            (r#"root ::= "a"; root ::= "b";"#.to_string(), "duplicate rule"),
            (r#"root ::= "a"**;"#.to_string(), "second quantifier"),
            (r#"root ::= [a"#.to_string(), "unterminated character class"),
            (r#"root ::= [z-a];"#.to_string(), "invalid class range"),
            (r#"root ::= 5;"#.to_string(), "unexpected byte"),
            (r#"root ::= "a" (* unterminated"#.to_string(), "unterminated comment"),
            (r#"root ::= [ab"#.to_string(), "unterminated character class"),
        ];
        for (text, needle) in cases {
            let err = compile_ebnf_nfa(text.as_str(), &GrammarLimits::default());
            assert!(err.is_err(), "expected error for: {text}");
            let msg = err.err().unwrap().to_string();
            assert!(msg.contains(needle), "message '{msg}' lacks '{needle}'");
        }
    }

    #[test]
    fn expression_nesting_is_bounded() {
        // 300 nested groups exceed the recursion bound of 256.
        let mut text = String::from("root ::= ");
        for _ in 0..300 {
            text.push('(');
        }
        text.push_str("\"x\"");
        for _ in 0..300 {
            text.push(')');
        }
        text.push(';');
        assert!(matches!(
            compile_ebnf_nfa(&text, &GrammarLimits::default()),
            Err(Error::CapacityExceeded(_))
        ));
        // 200 nested groups are fine.
        let mut ok = String::from("root ::= ");
        for _ in 0..200 {
            ok.push('(');
        }
        ok.push('"');
        ok.push('"');
        for _ in 0..200 {
            ok.push(')');
        }
        ok.push(';');
        let nfa = compile(&ok);
        assert!(accepted(&nfa, b""));
    }
}
