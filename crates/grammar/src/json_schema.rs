//! JSON Schema → NFA compiler (XGrammar's `json_schema_to_ebnf` path,
//! implemented as a direct NFA construction).
//!
//! Supported subset (anything else is a clear `Error::Grammar`, keeping
//! the compiler *sound* — a compiled grammar never accepts a document the
//! schema rejects): `type` (with type arrays), `properties` /
//! `required` / `additionalProperties` (bool or schema), `items` /
//! `prefixItems`, `minItems` / `maxItems`, `minLength` / `maxLength`,
//! `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`,
//! `enum` / `const`, `anyOf` / `oneOf`, `allOf` (DFA intersection),
//! `$ref` to `#/$defs/...` / `#/definitions/...` (cycle-free only),
//! boolean schemas (`true` = any, `false` = nothing).
//!
//! Unsupported keywords (sound rejection): `pattern`,
//! `patternProperties`, `format`, `not`, `if`/`then`/`else`,
//! `dependentSchemas` / `dependencies`, `propertyNames`, `contains`,
//! `uniqueItems`, `multipleOf`.
//!
//! Documented, soundness-preserving approximations:
//! * `minLength` / `maxLength` count **bytes**, not code points.
//! * Bounded numbers accept exact integers in range and plain fractional
//!   values whose integer part provably keeps the value in range, but
//!   reject exponent notation entirely (a strict subset of the valid
//!   forms — never a superset).
//! * One-sided integer bounds are capped at 2.3e18 (~19 digits).
//! * `oneOf` exclusivity is not context-free; it is compiled as `anyOf`
//!   (documented upstream limitation as well).

use crate::dfa;
use crate::ebnf::{lower_ast_no_rules, Ast};
use crate::nfa::{ClassSet, Fragment, Nfa, NfaBuilder};
use crate::GrammarLimits;
use deepinfer_core::json::Json;
use deepinfer_core::{ensure, Error, Result};
use std::collections::HashMap;

/// Maximum number of named properties in the any-order object product
/// construction (2^n mask states; beyond this the schema is rejected as
/// too complex — a capacity guard, not a semantic limit).
const MAX_NAMED_PROPERTIES: usize = 10;

/// Depth cap for the permissive "any JSON value" fallback (`items: true`,
/// `additionalProperties: true`, boolean `true` schemas). Each level
/// instantiates two fresh sub-automata (array element / object value), so
/// the cap bounds the expansion at 2^d instances; deeper values are
/// rejected (conservative but sound).
const MAX_ANY_DEPTH: u32 = 6;

/// Cap for one-sided integer bounds (~19 digits).
const INT_APPROX_MAX: i64 = i64::MAX / 4;

// ---------------------------------------------------------------------------
// Character classes / AST helpers
// ---------------------------------------------------------------------------

fn cls_digit() -> ClassSet {
    let mut c = ClassSet::empty();
    c.add_range(b'0', b'9');
    c
}

fn cls_hex() -> ClassSet {
    let mut c = cls_digit();
    c.add_range(b'a', b'f');
    c.add_range(b'A', b'F');
    c
}

fn cls_ws() -> ClassSet {
    let mut c = ClassSet::empty();
    for byte in [b' ', b'\t', b'\n', b'\r'] {
        c.add(byte);
    }
    c
}

/// Printable-ASCII JSON string body byte (no `"`, no `\`, no controls).
fn cls_plain_string_byte() -> ClassSet {
    let mut c = ClassSet::empty();
    c.add_range(0x20, 0x21);
    c.add_range(0x23, 0x5B);
    c.add_range(0x5D, 0x7E);
    c
}

fn lit(s: &str) -> Ast {
    Ast::Literal(s.as_bytes().to_vec())
}

fn class(c: ClassSet) -> Ast {
    Ast::Class(c)
}

fn seq(parts: Vec<Ast>) -> Ast {
    Ast::Seq(parts)
}

fn alt(parts: Vec<Ast>) -> Ast {
    Ast::Alt(parts)
}

fn repeat(min: u32, max: Option<u32>, inner: Ast) -> Ast {
    Ast::Repeat {
        min,
        max,
        inner: Box::new(inner),
    }
}

fn opt(inner: Ast) -> Ast {
    repeat(0, Some(1), inner)
}

fn star(inner: Ast) -> Ast {
    repeat(0, None, inner)
}

fn plus(inner: Ast) -> Ast {
    repeat(1, None, inner)
}

// ---------------------------------------------------------------------------
// JSON string grammar (strict-ish UTF-8 + escapes)
// ---------------------------------------------------------------------------

/// A single non-escape string byte: printable ASCII or a valid UTF-8
/// multi-byte sequence (surrogate encodings excluded).
fn string_char_ast() -> Ast {
    let plain = class(cls_plain_string_byte());
    let mut cont = ClassSet::empty();
    cont.add_range(0x80, 0xBF);
    let cont = class(cont);
    // 2-byte: C2..DF + cont
    let mut c2 = ClassSet::empty();
    c2.add_range(0xC2, 0xDF);
    let two = seq(vec![class(c2), cont.clone()]);
    // 3-byte: E0..EF minus surrogates (ED A0..BF), each + 2 continuations.
    let mut non_ed = ClassSet::empty();
    non_ed.add_range(0xE0, 0xEC);
    non_ed.add_range(0xEE, 0xEF);
    let mut ed = ClassSet::empty();
    ed.add(0xED);
    let mut ed_ok = ClassSet::empty();
    ed_ok.add_range(0x80, 0x9F);
    let three = alt(vec![
        seq(vec![class(non_ed), cont.clone(), cont.clone()]),
        seq(vec![class(ed), class(ed_ok), cont.clone()]),
    ]);
    // 4-byte: F0..F4 + 3 continuations.
    let mut c4 = ClassSet::empty();
    c4.add_range(0xF0, 0xF4);
    let four = seq(vec![class(c4), cont.clone(), cont.clone(), cont]);
    alt(vec![plain, two, three, four])
}

/// `"\uXXXX"` escapes with strict surrogate pairing.
fn unicode_escape_ast() -> Ast {
    let hex = class(cls_hex());
    // BMP escape excluding surrogates D800..DFFF.
    let mut not_d = cls_hex();
    not_d.remove(b'D');
    not_d.remove(b'd');
    let mut d = ClassSet::empty();
    d.add(b'D');
    d.add(b'd');
    let mut second_07 = ClassSet::empty();
    second_07.add_range(b'0', b'7');
    let single = alt(vec![
        seq(vec![class(not_d), hex.clone(), hex.clone()]),
        seq(vec![class(d.clone()), class(second_07), hex.clone(), hex.clone()]),
    ]);
    // Surrogate pair: \uD8..DB.. \uDC..DF..
    let mut second_hi = ClassSet::empty();
    second_hi.add_range(b'8', b'B');
    second_hi.add_range(b'b', b'b');
    let mut second_lo = ClassSet::empty();
    second_lo.add_range(b'C', b'F');
    second_lo.add_range(b'c', b'f');
    let pair = seq(vec![
        lit("\\u"),
        class(d),
        class(second_hi),
        hex.clone(),
        hex.clone(),
        lit("\\u"),
        class({
            let mut c = ClassSet::empty();
            c.add(b'D');
            c.add(b'd');
            c
        }),
        class(second_lo),
        hex.clone(),
        hex,
    ]);
    alt(vec![seq(vec![lit("\\u"), single]), pair])
}

fn string_body_ast() -> Ast {
    let escape = alt(vec![
        lit("\\\""),
        lit("\\\\"),
        lit("\\/"),
        lit("\\b"),
        lit("\\f"),
        lit("\\n"),
        lit("\\r"),
        lit("\\t"),
        unicode_escape_ast(),
    ]);
    alt(vec![string_char_ast(), escape])
}

/// Full JSON string (quoted); length bounds count bytes.
fn string_ast(min_len: u64, max_len: Option<u64>) -> Ast {
    let max = max_len.map(|m| m.min(4096) as u32);
    seq(vec![
        lit("\""),
        repeat(min_len.min(4096) as u32, max, string_body_ast()),
        lit("\""),
    ])
}

// ---------------------------------------------------------------------------
// Number grammar
// ---------------------------------------------------------------------------

fn integer_body_ast() -> Ast {
    let mut c19 = ClassSet::empty();
    c19.add_range(b'1', b'9');
    alt(vec![
        lit("0"),
        seq(vec![class(c19), star(class(cls_digit()))]),
    ])
}

fn number_body_ast() -> Ast {
    let mut ce = ClassSet::empty();
    ce.add(b'e');
    ce.add(b'E');
    let mut csign = ClassSet::empty();
    csign.add(b'+');
    csign.add(b'-');
    seq(vec![
        opt(lit("-")),
        integer_body_ast(),
        opt(seq(vec![lit("."), plus(class(cls_digit()))])),
        opt(seq(vec![
            class(ce),
            opt(class(csign)),
            plus(class(cls_digit())),
        ])),
    ])
}

// ---------------------------------------------------------------------------
// Integer range automaton (exact, for minimum/maximum)
// ---------------------------------------------------------------------------

fn digits_of(v: u64) -> Vec<u8> {
    v.to_string().into_bytes()
}

/// Unsigned integers `[lo, hi]` (inclusive, `lo >= 1`), no leading zeros.
fn pos_range_ast(lo: u64, hi: u64) -> Ast {
    debug_assert!(lo >= 1 && hi >= lo);
    let lo_s = digits_of(lo);
    let hi_s = digits_of(hi);
    if lo_s.len() == hi_s.len() {
        return same_len_range(&lo_s, &hi_s);
    }
    let mut branches = Vec::new();
    // Lowest length: [lo, 10^L - 1].
    if let Some(p) = 10u64.checked_pow(lo_s.len() as u32) {
        branches.push(same_len_range(&lo_s, &digits_of((p - 1).min(hi))));
    }
    // Middle lengths: first digit 1..9, rest free.
    let mut first = ClassSet::empty();
    first.add_range(b'1', b'9');
    let dig = class(cls_digit());
    for l in (lo_s.len() + 1)..hi_s.len() {
        branches.push(seq(vec![
            class(first),
            repeat((l - 1) as u32, Some((l - 1) as u32), dig.clone()),
        ]));
    }
    // Highest length: [10^(H-1), hi].
    let hi_bottom = 10u64
        .checked_pow((hi_s.len() - 1) as u32)
        .unwrap_or(1);
    branches.push(same_len_range(&digits_of(hi_bottom.max(1)), &hi_s));
    alt(branches)
}

/// Same-length digit ranges `[lo, hi]` (both length-L digit strings,
/// lexicographic == numeric at equal length). Recursive prefix splitting.
fn same_len_range(lo: &[u8], hi: &[u8]) -> Ast {
    debug_assert_eq!(lo.len(), hi.len());
    let l = lo.len();
    if l == 0 {
        return Ast::Empty;
    }
    if lo == hi {
        return Ast::Literal(lo.to_vec());
    }
    let mut branches = Vec::new();
    let mut j = 0usize;
    // Shared strict prefix while lo[j] == hi[j].
    while j < l && lo[j] == hi[j] && j < l - 1 {
        let sub = same_len_range(&lo[j + 1..], &hi[j + 1..]);
        let mut parts = vec![Ast::Literal(lo[..=j].to_vec())];
        parts.push(sub);
        branches.push(seq(parts));
        j += 1;
    }
    if j == l - 1 {
        // Final position with a free tail-free interval [lo[j], hi[j]].
        let mut c = ClassSet::empty();
        if lo[j] <= hi[j] {
            c.add_range(lo[j], hi[j]);
        }
        if !c.is_empty() {
            let mut parts = Vec::new();
            if j > 0 {
                parts.push(Ast::Literal(lo[..j].to_vec()));
            }
            parts.push(Ast::Class(c));
            branches.push(seq(parts));
        }
    } else if lo[j] < hi[j] {
        // Digit strictly between lo[j] and hi[j]: rest free.
        let tail = (l - j - 1) as u32;
        let mut mid = ClassSet::empty();
        if lo[j] + 1 <= hi[j] - 1 {
            mid.add_range(lo[j] + 1, hi[j] - 1);
        }
        if !mid.is_empty() {
            let mut parts = Vec::new();
            if j > 0 {
                parts.push(Ast::Literal(lo[..j].to_vec()));
            }
            parts.push(Ast::Class(mid));
            if tail > 0 {
                parts.push(repeat(tail, Some(tail), class(cls_digit())));
            }
            branches.push(seq(parts));
        }
        // lo-prefix branch: [lo[j]+1, 9] free tail.
        let mut lo_open = ClassSet::empty();
        if lo[j] < b'9' {
            lo_open.add_range(lo[j] + 1, b'9');
        }
        if !lo_open.is_empty() {
            let mut parts = Vec::new();
            if j > 0 {
                parts.push(Ast::Literal(lo[..j].to_vec()));
            }
            parts.push(Ast::Class(lo_open));
            if tail > 0 {
                parts.push(repeat(tail, Some(tail), class(cls_digit())));
            }
            branches.push(seq(parts));
        }
        // hi-prefix branch: [0, hi[j]-1] free tail.
        let mut hi_open = ClassSet::empty();
        if hi[j] > b'0' {
            hi_open.add_range(b'0', hi[j] - 1);
        }
        if !hi_open.is_empty() {
            let mut parts = Vec::new();
            if j > 0 {
                parts.push(Ast::Literal(lo[..j].to_vec()));
            }
            parts.push(Ast::Class(hi_open));
            if tail > 0 {
                parts.push(repeat(tail, Some(tail), class(cls_digit())));
            }
            branches.push(seq(parts));
        }
        // Strict-prefix continuation (lo[j] == hi[j] handled above).
    }
    if branches.is_empty() {
        return Ast::Literal(lo.to_vec());
    }
    alt(branches)
}

/// Signed integers in `[lo, hi]` (inclusive).
fn int_range_ast(lo: i64, hi: i64) -> Ast {
    if lo > hi {
        return Ast::Class(ClassSet::empty());
    }
    let mut branches = Vec::new();
    if lo <= 0 && hi >= 0 {
        branches.push(lit("0"));
    }
    if hi > 0 {
        branches.push(pos_range_ast(lo.max(1) as u64, hi as u64));
    }
    if lo < 0 {
        // -m in [lo, -1] iff m in [1, -lo].
        branches.push(seq(vec![lit("-"), pos_range_ast(1, (-(lo)) as u64)]));
    }
    alt(branches)
}

fn int_or_above(lo: i64) -> Ast {
    int_range_ast(lo, INT_APPROX_MAX)
}

fn int_or_below(hi: i64) -> Ast {
    int_range_ast(INT_APPROX_MIN, hi)
}

const INT_APPROX_MIN: i64 = -INT_APPROX_MAX;

// ---------------------------------------------------------------------------
// Schema compilation
// ---------------------------------------------------------------------------

pub struct SchemaCtx<'a> {
    root: &'a Json,
    limits: &'a GrammarLimits,
    ref_stack: Vec<String>,
}

/// Compiles a JSON Schema into an NFA.
pub fn compile_json_schema_nfa(schema: &Json, limits: &GrammarLimits) -> Result<Nfa> {
    let ctx = SchemaCtx {
        root: schema,
        limits,
        ref_stack: Vec::new(),
    };
    let mut builder = NfaBuilder::new(limits.max_nfa_nodes);
    let frag = schema_fragment(&mut builder, &ctx, schema)?;
    builder.finish(frag)
}

fn unsupported(what: &str) -> Error {
    Error::Grammar(format!("unsupported JSON Schema keyword: {what}"))
}

fn ws_frag(b: &mut NfaBuilder, limits: &GrammarLimits) -> Result<Option<Fragment>> {
    lower_ast_no_rules(b, &star(class(cls_ws())), limits)
}

/// Like [`ws_frag`] but guaranteed present (whitespace star can never be
/// the empty language); used where the wiring needs a concrete node.
fn ws_node(b: &mut NfaBuilder, limits: &GrammarLimits) -> Result<Fragment> {
    ws_frag(b, limits)?
        .ok_or_else(|| Error::Internal("whitespace fragment unexpectedly dead".into()))
}

fn schema_fragment(
    b: &mut NfaBuilder,
    ctx: &SchemaCtx<'_>,
    schema: &Json,
) -> Result<Option<Fragment>> {
    if let Some(v) = schema.as_bool() {
        return if v {
            any_json_fragment(b, ctx.limits, MAX_ANY_DEPTH)
        } else {
            Ok(None)
        };
    }
    let obj: HashMap<String, &Json> = schema
        .entries()
        .iter()
        .map(|(k, v)| (k.clone(), v))
        .collect();
    for key in [
        "pattern",
        "patternProperties",
        "format",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "dependencies",
        "propertyNames",
        "contains",
        "uniqueItems",
        "multipleOf",
    ] {
        if obj.contains_key(key) {
            return Err(unsupported(key));
        }
    }
    if let Some(r) = obj.get("$ref").and_then(|v| v.as_str()) {
        let sibling_free = obj
            .keys()
            .all(|k| matches!(k.as_str(), "$ref" | "$defs" | "definitions"));
        ensure!(
            sibling_free,
            Grammar,
            "$ref must stand alone (siblings besides $defs are unsupported): {r:?}"
        );
        let target = resolve_ref(ctx, r)?;
        ensure!(
            !ctx.ref_stack.iter().any(|s| s == r),
            Grammar,
            "cyclic $ref {r:?} cannot terminate"
        );
        let mut sub_ctx = SchemaCtx {
            root: ctx.root,
            limits: ctx.limits,
            ref_stack: ctx.ref_stack.clone(),
        };
        sub_ctx.ref_stack.push(r.to_string());
        return schema_fragment(b, &sub_ctx, target);
    }
    if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_arr()) {
        ensure!(!all_of.is_empty(), Grammar, "allOf must be non-empty");
        let mut dfas = Vec::with_capacity(all_of.len());
        for sub in all_of {
            let nfa = {
                let sub_ctx = SchemaCtx {
                    root: ctx.root,
                    limits: ctx.limits,
                    ref_stack: ctx.ref_stack.clone(),
                };
                let mut sb = NfaBuilder::new(ctx.limits.max_nfa_nodes);
                let frag = schema_fragment(&mut sb, &sub_ctx, sub)?;
                sb.finish(frag)?
            };
            let d = dfa::determinize(&nfa, ctx.limits.max_dfa_states)?;
            dfas.push(dfa::prune_live(d));
        }
        let product = dfa::intersect_all(dfas, ctx.limits.max_dfa_states)?;
        return dfa::embed_dfa(b, &product).map(Some);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(arr) = obj.get(key).and_then(|v| v.as_arr()) {
            ensure!(!arr.is_empty(), Grammar, "{key} must be non-empty");
            let mut branches = Vec::with_capacity(arr.len());
            for sub in arr {
                branches.push(schema_fragment(b, ctx, sub)?);
            }
            return b.alt(branches);
        }
    }
    if let Some(arr) = obj.get("enum").and_then(|v| v.as_arr()) {
        ensure!(!arr.is_empty(), Grammar, "enum must be non-empty");
        let mut branches = Vec::with_capacity(arr.len());
        for v in arr {
            branches.push(b.frag_literal(&canonical_json(v)?).map(Some)?);
        }
        return b.alt(branches);
    }
    if let Some(v) = obj.get("const") {
        return b.frag_literal(&canonical_json(v)?).map(Some);
    }
    let type_names = collect_types(&obj);
    if type_names.is_empty() {
        return any_json_fragment(b, ctx.limits, MAX_ANY_DEPTH);
    }
    // Scalar types compile through the AST; containers build NFAs directly.
    let min_len = obj.get("minLength").and_then(|v| v.as_u64()).unwrap_or(0);
    let max_len = obj.get("maxLength").and_then(|v| v.as_u64());
    let minimum = numeric_bound(&obj, "minimum", "exclusiveMinimum", false);
    let maximum = numeric_bound(&obj, "maximum", "exclusiveMaximum", true);

    let mut scalar_branches: Vec<Option<Fragment>> = Vec::new();
    let mut needs_container = false;
    for t in &type_names {
        match t.as_str() {
            "string" => {
                let ast = string_ast(min_len, max_len);
                scalar_branches.push(lower_ast_no_rules(b, &ast, ctx.limits)?);
            }
            "integer" => {
                let ast = match (minimum, maximum) {
                    (Some(lo), Some(hi)) => {
                        int_range_ast(bound_i64(lo, true), bound_i64(hi, false))
                    }
                    (Some(lo), None) => int_or_above(bound_i64(lo, true)),
                    (None, Some(hi)) => int_or_below(bound_i64(hi, false)),
                    (None, None) => seq(vec![opt(lit("-")), integer_body_ast()]),
                };
                scalar_branches.push(lower_ast_no_rules(b, &ast, ctx.limits)?);
            }
            "number" => {
                let ast = match (minimum, maximum) {
                    (Some(lo), Some(hi)) => bounded_number_ast(lo, hi),
                    (Some(lo), None) => number_with_min(lo),
                    (None, Some(hi)) => number_with_max(hi),
                    (None, None) => number_body_ast(),
                };
                scalar_branches.push(lower_ast_no_rules(b, &ast, ctx.limits)?);
            }
            "boolean" => {
                scalar_branches.push(b.frag_literal(b"true").map(Some)?);
                scalar_branches.push(b.frag_literal(b"false").map(Some)?);
            }
            "null" => scalar_branches.push(b.frag_literal(b"null").map(Some)?),
            "array" => needs_container = true,
            "object" => needs_container = true,
            other => {
                return Err(Error::Grammar(format!("unsupported JSON type {other:?}")));
            }
        }
    }
    if needs_container {
        let mut branches = scalar_branches;
        if type_names.iter().any(|t| t == "array") {
            branches.push(array_fragment(b, ctx, &obj)?);
        }
        if type_names.iter().any(|t| t == "object") {
            branches.push(object_fragment(b, ctx, &obj)?);
        }
        return b.alt(branches);
    }
    b.alt(scalar_branches)
}

fn collect_types(obj: &HashMap<String, &Json>) -> Vec<String> {
    if let Some(t) = obj.get("type") {
        if let Some(s) = t.as_str() {
            return vec![s.to_string()];
        }
        if let Some(arr) = t.as_arr() {
            return arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
        }
    }
    Vec::new()
}

fn numeric_bound(
    obj: &HashMap<String, &Json>,
    plain: &str,
    exclusive: &str,
    upper: bool,
) -> Option<f64> {
    let e = obj.get(exclusive).and_then(|v| v.as_f64());
    let p = obj.get(plain).and_then(|v| v.as_f64());
    match (p, e) {
        (None, None) => None,
        (Some(p), None) => Some(p),
        (None, Some(e)) => Some(if upper { e - 1e-9 } else { e + 1e-9 }),
        (Some(_), Some(e)) => Some(if upper { e - 1e-9 } else { e + 1e-9 }),
    }
}

fn bound_i64(v: f64, lower: bool) -> i64 {
    if lower {
        v.ceil() as i64
    } else {
        v.floor() as i64
    }
}

/// Bounded number: exact integers in range, plus plain fractional values
/// whose integer part provably keeps the value inside the bounds. No
/// exponents (documented sound subset).
fn bounded_number_ast(lo: f64, hi: f64) -> Ast {
    let lo_i = bound_i64(lo, true);
    let hi_i = bound_i64(hi, false);
    let mut branches = Vec::new();
    if lo_i <= hi_i {
        branches.push(int_range_ast(lo_i, hi_i));
    }
    // Positive fractional: value = p + f, f in (0,1). Valid iff p >= lo
    // and p + 1 <= hi  =>  p in [ceil(lo), floor(hi - 1)].
    let p_lo = lo.ceil() as i64;
    let p_hi = (hi - 1.0).floor() as i64;
    if p_lo >= 0 && p_hi >= p_lo {
        branches.push(seq(vec![
            pos_or_range(p_lo as u64, p_hi as u64),
            lit("."),
            plus(class(cls_digit())),
        ]));
    }
    // Negative fractional: value = -(p + f). Valid iff -(p+f) >= lo and
    // -(p+f) <= hi  =>  p + f <= -lo and p + f >= -hi.
    // p <= ceil(-lo) - 1 and p >= max(0, ceil(-hi)).
    let neg_lo = (-hi).ceil().max(0.0) as i64; // p lower bound
    let neg_hi = (-(lo)).ceil() as i64 - 1; // p upper bound
    if neg_hi >= neg_lo && neg_hi >= 0 {
        branches.push(seq(vec![
            lit("-"),
            pos_or_range(neg_lo.max(0) as u64 + u64::from(neg_lo < 1), neg_hi as u64),
            lit("."),
            plus(class(cls_digit())),
        ]));
    }
    if branches.is_empty() {
        return Ast::Class(ClassSet::empty());
    }
    alt(branches)
}

fn pos_or_range(lo: u64, hi: u64) -> Ast {
    if lo <= 1 {
        integer_body_ast()
    } else {
        pos_range_ast(lo, hi.max(lo))
    }
}

fn number_with_min(lo: f64) -> Ast {
    let mut branches = Vec::new();
    branches.push(int_or_above(bound_i64(lo, true)));
    // Positive fractional: p >= ceil(lo) (or any p >= 0 when lo <= 0).
    if lo > 0.0 {
        branches.push(seq(vec![
            pos_or_range(lo.ceil() as u64, INT_APPROX_MAX as u64),
            lit("."),
            plus(class(cls_digit())),
        ]));
    } else {
        branches.push(seq(vec![
            integer_body_ast(),
            lit("."),
            plus(class(cls_digit())),
        ]));
        // Negative fractional: -(p+f) >= lo => p <= ceil(-lo) - 1.
        let neg_hi = (-(lo)).ceil() as i64 - 1;
        if neg_hi >= 0 {
            branches.push(seq(vec![
                lit("-"),
                pos_or_range(0, neg_hi as u64),
                lit("."),
                plus(class(cls_digit())),
            ]));
        }
    }
    alt(branches)
}

fn number_with_max(hi: f64) -> Ast {
    let mut branches = Vec::new();
    branches.push(int_or_below(bound_i64(hi, false)));
    // Positive fractional: p + f <= hi => p <= ceil(hi) - 1.
    let p_hi = hi.ceil() as i64 - 1;
    if p_hi >= 0 {
        branches.push(seq(vec![
            pos_or_range(0, p_hi as u64),
            lit("."),
            plus(class(cls_digit())),
        ]));
    }
    // Negative fractional: -(p+f) <= hi. When hi >= 0 always true.
    if hi >= 0.0 {
        branches.push(seq(vec![
            lit("-"),
            integer_body_ast(),
            lit("."),
            plus(class(cls_digit())),
        ]));
    }
    alt(branches)
}

// ---------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------

fn array_fragment(
    b: &mut NfaBuilder,
    ctx: &SchemaCtx<'_>,
    obj: &HashMap<String, &Json>,
) -> Result<Option<Fragment>> {
    let min_items = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let max_items = obj.get("maxItems").and_then(|v| v.as_u64()).map(|v| v as u32);
    let prefix = obj.get("prefixItems").and_then(|v| v.as_arr());
    let n_prefix = prefix.map(|p| p.len()).unwrap_or(0) as u32;
    let items = obj.get("items");

    // Element fragment for positions >= n_prefix.
    let tail_item: Option<Fragment> = match items {
        None if n_prefix == 0 => any_json_fragment(b, ctx.limits, MAX_ANY_DEPTH)?,
        None => None,
        Some(Json::Bool(true)) => any_json_fragment(b, ctx.limits, MAX_ANY_DEPTH)?,
        Some(Json::Bool(false)) => None,
        Some(sub) => schema_fragment(b, ctx, sub)?,
    };

    let required = min_items.max(n_prefix);
    if let Some(mx) = max_items {
        if mx < required {
            return Ok(None);
        }
    }
    if tail_item.is_none() && required > n_prefix {
        return Ok(None);
    }

    let open = b.frag_literal(b"[")?;
    let close = b.frag_literal(b"]")?;
    let ws1 = ws_frag(b, ctx.limits)?;
    let ws2 = ws_frag(b, ctx.limits)?;
    let comma = b.frag_literal(b",")?;
    let ws3 = ws_frag(b, ctx.limits)?;
    let sep: Option<Fragment> = b.seq(vec![ws2, Some(comma), ws3])?; // ws , ws

    // parts: '[' ws (elem (sep elem)*)? ws ']'
    let mut parts: Vec<Option<Fragment>> = vec![Some(open), ws1];
    if required == 0 {
        if let Some(item) = tail_item {
            parts.push(Some(b.optional(item)?));
            let si = b.seq(vec![sep, Some(item)])?;
            if let Some(si) = si {
                parts.push(Some(b.star(si)?));
            }
        }
    } else {
        for i in 0..required {
            if i > 0 {
                parts.push(sep);
            }
            let f = if i < n_prefix {
                schema_fragment(b, ctx, &prefix.unwrap()[i as usize])?
            } else {
                tail_item
            };
            let Some(f) = f else { return Ok(None) };
            parts.push(Some(f));
        }
        if let Some(item) = tail_item {
            match max_items {
                None => {
                    let si = b.seq(vec![sep, Some(item)])?;
                    if let Some(si) = si {
                        parts.push(Some(b.star(si)?));
                    }
                }
                Some(mx) => {
                    for _ in required..mx {
                        let si = b.seq(vec![sep, Some(item)])?;
                        if let Some(si) = si {
                            parts.push(Some(b.optional(si)?));
                        }
                    }
                }
            }
        }
    }
    let ws4 = ws_frag(b, ctx.limits)?;
    parts.push(ws4);
    parts.push(Some(close));
    b.seq(parts)
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

fn object_fragment(
    b: &mut NfaBuilder,
    ctx: &SchemaCtx<'_>,
    obj: &HashMap<String, &Json>,
) -> Result<Option<Fragment>> {
    let props: Vec<(String, &Json)> = obj
        .get("properties")
        .map(|p| p.entries().iter().map(|(k, v)| (k.clone(), v)).collect())
        .unwrap_or_default();
    let required: Vec<String> = obj
        .get("required")
        .and_then(|v| v.as_arr())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let additional = obj.get("additionalProperties");

    ensure!(
        props.len() <= MAX_NAMED_PROPERTIES,
        CapacityExceeded,
        "object has {} named properties; the any-order product construction supports at most {MAX_NAMED_PROPERTIES}",
        props.len()
    );
    // Every required property must exist in `properties`.
    for r in &required {
        ensure!(
            props.iter().any(|(k, _)| k == r),
            Grammar,
            "required property {r:?} has no schema under 'properties'"
        );
    }
    let required_mask: u32 = props
        .iter()
        .enumerate()
        .filter(|(_, (k, _))| required.contains(k))
        .map(|(i, _)| 1u32 << i)
        .sum();

    let entry = b.frag_literal(b"{")?;
    let end = b.node()?;
    let ws_open = ws_node(b, ctx.limits)?;
    let init = b.node()?; // I: object content start (mask 0)

    // Junction states, mirroring the JSON grammar exactly:
    //   I  — after '{' ws (or object start): pairs may begin; '}' allowed
    //        only when nothing is required.
    //   R_m — right after a completed pair with used-mask m: '}' allowed
    //        when required bits are all set; ',' leads to P_m.
    //   P_m — after a comma: a pair MUST follow (no '}' — trailing
    //        commas are invalid JSON).
    let mut r_nodes: HashMap<u32, u32> = HashMap::new();
    let mut p_nodes: HashMap<u32, u32> = HashMap::new();

    let r_node = |b: &mut NfaBuilder, m: u32, r_nodes: &mut HashMap<u32, u32>| -> Result<u32> {
        match r_nodes.get(&m) {
            Some(&n) => Ok(n),
            None => {
                let n = b.node()?;
                r_nodes.insert(m, n);
                Ok(n)
            }
        }
    };
    let p_node = |b: &mut NfaBuilder, m: u32, p_nodes: &mut HashMap<u32, u32>| -> Result<u32> {
        match p_nodes.get(&m) {
            Some(&n) => Ok(n),
            None => {
                let n = b.node()?;
                p_nodes.insert(m, n);
                Ok(n)
            }
        }
    };

    /// Builds one `key : value` pair fragment (fresh wiring per call so
    /// each use site owns its continuation).
    fn build_pair(
        b: &mut NfaBuilder,
        ctx: &SchemaCtx<'_>,
        key_lit: &str,
        sub: &Json,
    ) -> Result<Option<Fragment>> {
        let parts: Vec<Option<Fragment>> = vec![
            b.frag_literal(key_lit.as_bytes()).map(Some)?,
            ws_frag(b, ctx.limits)?,
            b.frag_literal(b":").map(Some)?,
            ws_frag(b, ctx.limits)?,
            schema_fragment(b, ctx, sub)?,
            ws_frag(b, ctx.limits)?,
        ];
        b.seq(parts)
    }

    // Deterministic BFS over junction states: (kind, mask) sorted
    // ascending; kind: 0 = I, 1 = R, 2 = P.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Kind {
        Init,
        AfterPair,
        AfterComma,
    }
    let mut queue: Vec<(Kind, u32)> = vec![(Kind::Init, 0)];
    let mut seen: Vec<(Kind, u32)> = vec![(Kind::Init, 0)];

    while let Some(idx) = queue
        .iter()
        .enumerate()
        .min_by_key(|(_, &st)| st)
        .map(|(i, _)| i)
    {
        let (kind, m) = queue.swap_remove(idx);
        let node = match kind {
            Kind::Init => init,
            Kind::AfterPair => r_nodes[&m],
            Kind::AfterComma => p_nodes[&m],
        };
        // Pairs can start from I and P states.
        if matches!(kind, Kind::Init | Kind::AfterComma) {
            for (i, (key, sub)) in props.iter().enumerate() {
                let bit = 1u32 << i;
                if m & bit != 0 {
                    continue;
                }
                let key_lit = format!("\"{}\"", escape_json_string(key));
                let Some(pair) = build_pair(b, ctx, &key_lit, sub)? else {
                    continue;
                };
                let next_m = m | bit;
                let target = r_node(b, next_m, &mut r_nodes)?;
                b.edge_eps(node, pair.start);
                b.edge_eps(pair.end, target);
                if !seen.contains(&(Kind::AfterPair, next_m)) {
                    seen.push((Kind::AfterPair, next_m));
                    queue.push((Kind::AfterPair, next_m));
                }
            }
            if !matches!(additional, Some(Json::Bool(false))) {
                let generic_value: Option<Fragment> = match additional {
                    Some(Json::Bool(true)) | None => {
                        any_json_fragment(b, ctx.limits, MAX_ANY_DEPTH)?
                    }
                    Some(sub) => schema_fragment(b, ctx, sub)?,
                };
                if let Some(value_frag) = generic_value {
                    let key_ast = if props.is_empty() {
                        string_ast(0, None)
                    } else {
                        exclude_keys_ast(
                            &props.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
                        )
                    };
                    let key_frag = lower_ast_no_rules(b, &key_ast, ctx.limits)?;
                    let parts: Vec<Option<Fragment>> = vec![
                        key_frag,
                        ws_frag(b, ctx.limits)?,
                        b.frag_literal(b":").map(Some)?,
                        ws_frag(b, ctx.limits)?,
                        Some(value_frag),
                        ws_frag(b, ctx.limits)?,
                    ];
                    if let Some(pair) = b.seq(parts)? {
                        let target = r_node(b, m, &mut r_nodes)?;
                        b.edge_eps(node, pair.start);
                        b.edge_eps(pair.end, target);
                        if !seen.contains(&(Kind::AfterPair, m)) {
                            seen.push((Kind::AfterPair, m));
                            queue.push((Kind::AfterPair, m));
                        }
                    }
                }
            }
            // '}' from I only when nothing is required (empty object).
            if matches!(kind, Kind::Init) && required_mask == 0 {
                let close = b.frag_literal(b"}")?;
                b.edge_eps(node, close.start);
                b.edge_eps(close.end, end);
            }
        }
        // After a pair: '}' (when required satisfied) or ',' -> P_m.
        if matches!(kind, Kind::AfterPair) {
            if (m & required_mask) == required_mask {
                let close = b.frag_literal(b"}")?;
                b.edge_eps(node, close.start);
                b.edge_eps(close.end, end);
            }
            let comma = b.frag_literal(b",")?;
            let ws_c = ws_node(b, ctx.limits)?;
            let target = p_node(b, m, &mut p_nodes)?;
            b.edge_eps(node, comma.start);
            b.edge_eps(comma.end, ws_c.start);
            b.edge_eps(ws_c.end, target);
            if !seen.contains(&(Kind::AfterComma, m)) {
                seen.push((Kind::AfterComma, m));
                queue.push((Kind::AfterComma, m));
            }
        }
    }

    b.edge_eps(entry.end, ws_open.start);
    b.edge_eps(ws_open.start, init);
    Ok(Some(Fragment {
        start: entry.start,
        end,
    }))
}

/// A JSON string (quoted) whose content is NOT byte-equal to any of
/// `keys` (ASCII keys only; non-ASCII keys make the exclusion
/// inexact-but-sound by rejecting that length class of lookalikes is NOT
/// attempted — instead non-ASCII keys disable the generic path via an
/// error at a higher level is avoided by accepting all strings of other
/// shapes; see docs). Lengths beyond the longest key are always allowed.
fn exclude_keys_ast(keys: &[String]) -> Ast {
    let body = string_body_ast();
    let max_len = keys.iter().map(|k| k.len()).max().unwrap_or(0);
    let mut branches = Vec::new();
    for l in 0..=max_len {
        let mut diffs: Vec<Ast> = Vec::new();
        let mut has_non_ascii = false;
        for key in keys {
            if key.len() != l {
                continue;
            }
            if !key.is_ascii() {
                has_non_ascii = true;
                continue;
            }
            for pos in 0..l {
                let mut parts: Vec<Ast> = Vec::new();
                for ch in &key.as_bytes()[..pos] {
                    parts.push(Ast::Literal(vec![*ch]));
                }
                // A body unit differing from key[pos]: multi-byte units
                // never equal an ASCII byte, so excluding that byte from
                // the 1-byte class and allowing any other unit works.
                let mut c = cls_plain_string_byte();
                c.remove(key.as_bytes()[pos]);
                parts.push(Ast::Class(c));
                for _ in (pos + 1)..l {
                    parts.push(body.clone());
                }
                diffs.push(seq(parts));
            }
        }
        if has_non_ascii {
            // Cannot exclude precisely: allow all strings of this length
            // (a sound superset is NOT acceptable; instead this branch
            // accepts only strings that differ from ASCII keys, and
            // non-ASCII keys of this length are conservatively EXCLUDED
            // by requiring at least one difference from some key).
            // Simplest sound fallback: allow the diff branches only.
            if diffs.is_empty() {
                // Nothing safe to allow at this length: skip it.
                continue;
            }
        }
        if diffs.is_empty() {
            branches.push(repeat(l as u32, Some(l as u32), body.clone()));
        } else {
            branches.push(alt(diffs));
        }
    }
    // Longer than every key: always fine.
    branches.push(seq(vec![
        repeat(0, Some(max_len as u32), body.clone()),
        plus(body),
    ]));
    seq(vec![lit("\""), alt(branches), lit("\"")])
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Canonical (minimal) JSON encoding for `enum` / `const` scalars.
fn canonical_json(v: &Json) -> Result<Vec<u8>> {
    match v {
        Json::Null => Ok(b"null".to_vec()),
        Json::Bool(true) => Ok(b"true".to_vec()),
        Json::Bool(false) => Ok(b"false".to_vec()),
        Json::U64(n) => Ok(n.to_string().into_bytes()),
        Json::I64(n) => Ok(n.to_string().into_bytes()),
        Json::F64(n) => {
            ensure!(
                n.fract() == 0.0 && n.abs() < 9e15,
                Grammar,
                "enum/const numbers must be integers (got {n})"
            );
            Ok(format!("{}", *n as i64).into_bytes())
        }
        Json::Str(s) => Ok(format!("\"{}\"", escape_json_string(s)).into_bytes()),
        other => Err(Error::Grammar(format!(
            "enum/const values must be scalars, got {other:?}"
        ))),
    }
}

/// Depth-capped "any JSON value" (scalars + nested arrays/objects).
///
/// Each nesting level instantiates fresh sub-automata for the array
/// element and the object value contexts (sharing a fragment across
/// contexts would be unsound), so the depth cap bounds the node count at
/// `O(2^depth)`; `MAX_ANY_DEPTH = 6` keeps that well inside the node
/// budget.
fn any_json_fragment(
    b: &mut NfaBuilder,
    limits: &GrammarLimits,
    depth: u32,
) -> Result<Option<Fragment>> {
    if depth == 0 {
        let scalars: Vec<Option<Fragment>> = vec![
            b.frag_literal(b"null").map(Some)?,
            b.frag_literal(b"true").map(Some)?,
            b.frag_literal(b"false").map(Some)?,
            lower_ast_no_rules(b, &string_ast(0, None), limits)?,
            lower_ast_no_rules(b, &number_body_ast(), limits)?,
        ];
        return b.alt(scalars);
    }
    // Array: '[' ws list ']' with list = (elem (',' elem)*)? built as a
    // single-instance loop: loop -> elem -> ',' -> loop, loop -> exit.
    let arr = {
        let open = b.frag_literal(b"[")?;
        let close = b.frag_literal(b"]")?;
        let ws1 = ws_frag(b, limits)?;
        let ws2 = ws_frag(b, limits)?;
        let elem = any_json_fragment(b, limits, depth - 1)?;
        let list = match elem {
            Some(e) => comma_loop(b, e)?,
            None => b.frag_empty()?,
        };
        b.seq(vec![Some(open), ws1, Some(list), ws2, Some(close)])?
    };
    // Object: '{' ws pair-list '}' with pair = string ws ':' ws value and
    // the same single-instance comma loop over pairs.
    let obj = {
        let obrace = b.frag_literal(b"{")?;
        let cbrace = b.frag_literal(b"}")?;
        let ws1 = ws_frag(b, limits)?;
        let ws2 = ws_frag(b, limits)?;
        let key = lower_ast_no_rules(b, &string_ast(0, None), limits)?;
        let colon = b.frag_literal(b":")?;
        let ws3 = ws_frag(b, limits)?;
        let value = any_json_fragment(b, limits, depth - 1)?;
        let pair = match value {
            Some(v) => b.seq(vec![key, ws3, Some(colon), Some(v)])?,
            None => None,
        };
        let list = match pair {
            Some(p) => comma_loop(b, p)?,
            None => b.frag_empty()?,
        };
        b.seq(vec![Some(obrace), ws1, Some(list), ws2, Some(cbrace)])?
    };
    let all: Vec<Option<Fragment>> = vec![
        b.frag_literal(b"null").map(Some)?,
        b.frag_literal(b"true").map(Some)?,
        b.frag_literal(b"false").map(Some)?,
        lower_ast_no_rules(b, &string_ast(0, None), limits)?,
        lower_ast_no_rules(b, &number_body_ast(), limits)?,
        arr,
        obj,
    ];
    b.alt(all)
}

/// A `(item (',' item)*)?` loop re-using ONE instance of `item`:
/// `loop -> item.start`, `item.end -> comma`, `comma.end -> loop`, and the
/// fragment's start==end==loop node (so sequencing wires the exit).
fn comma_loop(b: &mut NfaBuilder, item: Fragment) -> Result<Fragment> {
    let comma = b.frag_literal(b",")?;
    let l = b.node()?;
    b.edge_eps(l, item.start);
    b.edge_eps(item.end, comma.start);
    b.edge_eps(comma.end, l);
    Ok(Fragment { start: l, end: l })
}

fn resolve_ref<'a>(ctx: &SchemaCtx<'a>, r: &str) -> Result<&'a Json> {
    let (container_key, name) = r
        .strip_prefix("#/$defs/")
        .map(|n| ("$defs", n))
        .or_else(|| r.strip_prefix("#/definitions/").map(|n| ("definitions", n)))
        .ok_or_else(|| {
            Error::Grammar(format!(
                "only local refs '#/$defs/...' and '#/definitions/...' are supported, got {r:?}"
            ))
        })?;
    ctx.root
        .get(container_key)
        .and_then(|d| d.get(name))
        .ok_or_else(|| Error::Grammar(format!("$ref target {r:?} not found")))
}
