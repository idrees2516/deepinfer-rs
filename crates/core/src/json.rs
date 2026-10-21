//! Minimal, allocation-bounded JSON parser and serializer.
//!
//! Used for checkpoint `config.json`, safetensors headers, tokenizer
//! configs and structured-output `so_args`. Hardened against
//! denial-of-service: recursion depth and nesting are bounded, all
//! indexing is checked, and the parser never panics.
use crate::ensure;


use crate::error::{Error, Result};
use std::fmt::Write as _;

/// Hard nesting limit (defensive bound against malicious inputs).
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value. Object key order is preserved.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// Integral number that fits `u64`.
    U64(u64),
    /// Integral number that fits `i64` (negative).
    I64(i64),
    /// Any other number.
    F64(f64),
    /// UTF-8 string.
    Str(String),
    /// Array.
    Arr(Vec<Json>),
    /// Object (ordered).
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses `text` (whole input must be valid JSON).
    pub fn parse(text: &str) -> Result<Json> {
        let mut p = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        p.skip_ws();
        let v = p.value(0)?;
        p.skip_ws();
        ensure!(
            p.pos == p.bytes.len(),
            Checkpoint,
            "trailing characters at offset {}",
            p.pos
        );
        Ok(v)
    }

    /// Looks up `key` in an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Object entries (empty for non-objects).
    pub fn entries(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(pairs) => pairs,
            _ => &[],
        }
    }

    /// Typed accessors.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Typed accessor for booleans.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Numeric accessor (integral values widen losslessly).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::U64(v) => Some(*v as f64),
            Json::I64(v) => Some(*v as f64),
            Json::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Unsigned integer accessor (rejects negatives / fractionals).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::U64(v) => Some(*v),
            Json::I64(v) if *v >= 0 => Some(*v as u64),
            Json::F64(v) if *v >= 0.0 && v.fract() == 0.0 && *v <= u64::MAX as f64 => Some(*v as u64),
            _ => None,
        }
    }

    /// Signed integer accessor.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::U64(v) if *v <= i64::MAX as u64 => Some(*v as i64),
            Json::I64(v) => Some(*v),
            Json::F64(v) if v.fract() == 0.0 && v.abs() <= i64::MAX as f64 => Some(*v as i64),
            _ => None,
        }
    }

    /// Array accessor.
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// Convenience: `self.get(key)` as `&str`.
    pub fn str_field(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Json::as_str)
    }

    /// Convenience: `self.get(key)` as `u64`.
    pub fn u64_field(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Json::as_u64)
    }

    /// Convenience: `self.get(key)` as `f64`.
    pub fn f64_field(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Json::as_f64)
    }

    /// Convenience: `self.get(key)` as `bool`.
    pub fn bool_field(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Json::as_bool)
    }

    /// Serializes to compact JSON text.
    pub fn to_json_string(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::U64(v) => {
                let _ = write!(out, "{v}");
            }
            Json::I64(v) => {
                let _ = write!(out, "{v}");
            }
            Json::F64(v) => {
                if v.is_finite() {
                    let _ = write!(out, "{v}");
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_escaped(out, s),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_escaped(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        ensure!(
            self.peek() == Some(b),
            Checkpoint,
            "expected '{}' at offset {}",
            b as char,
            self.pos
        );
        self.pos += 1;
        Ok(())
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        ensure!(depth <= MAX_DEPTH, CapacityExceeded, "json nesting exceeds {MAX_DEPTH}");
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal(b"true", Json::Bool(true)),
            Some(b'f') => self.literal(b"false", Json::Bool(false)),
            Some(b'n') => self.literal(b"null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(Error::Checkpoint(format!(
                "unexpected byte '{}' at offset {}",
                c as char, self.pos
            ))),
            None => Err(Error::Checkpoint("unexpected end of json".into())),
        }
    }

    fn literal(&mut self, lit: &[u8], v: Json) -> Result<Json> {
        ensure!(
            self.bytes.len() - self.pos >= lit.len() && &self.bytes[self.pos..self.pos + lit.len()] == lit,
            Checkpoint,
            "invalid literal at offset {}",
            self.pos
        );
        self.pos += lit.len();
        Ok(v)
    }

    fn object(&mut self, depth: usize) -> Result<Json> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let val = self.value(depth + 1)?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => {
                    return Err(Error::Checkpoint(format!(
                        "expected ',' or '}}' at offset {}",
                        self.pos
                    )))
                }
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => {
                    return Err(Error::Checkpoint(format!(
                        "expected ',' or ']' at offset {}",
                        self.pos
                    )))
                }
            }
        }
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let b = self
                .peek()
                .ok_or_else(|| Error::Checkpoint("unterminated string".into()))?;
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| Error::Checkpoint("unterminated escape".into()))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: expect a low surrogate.
                                ensure!(
                                    self.peek() == Some(b'\\'),
                                    Checkpoint,
                                    "lone high surrogate at offset {}",
                                    self.pos
                                );
                                self.pos += 1;
                                ensure!(self.peek() == Some(b'u'), Checkpoint, "expected low surrogate");
                                self.pos += 1;
                                let lo = self.hex4()?;
                                ensure!(
                                    (0xDC00..0xE000).contains(&lo),
                                    Checkpoint,
                                    "invalid low surrogate"
                                );
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(
                                    char::from_u32(c)
                                        .ok_or_else(|| Error::Checkpoint("invalid codepoint".into()))?,
                                );
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return Err(Error::Checkpoint("lone low surrogate".into()));
                            } else {
                                out.push(
                                    char::from_u32(cp)
                                        .ok_or_else(|| Error::Checkpoint("invalid codepoint".into()))?,
                                );
                            }
                        }
                        other => {
                            return Err(Error::Checkpoint(format!(
                                "invalid escape '\\{}'",
                                other as char
                            )))
                        }
                    }
                }
                _ => {
                    // Re-assemble UTF-8 multibyte sequences verbatim.
                    if b < 0x80 {
                        out.push(b as char);
                    } else {
                        let len = if b >= 0xF0 {
                            4
                        } else if b >= 0xE0 {
                            3
                        } else {
                            2
                        };
                        ensure!(
                            self.pos + len - 1 <= self.bytes.len(),
                            Checkpoint,
                            "truncated utf-8 in string"
                        );
                        let s = std::str::from_utf8(&self.bytes[self.pos - 1..self.pos + len - 1])
                            .map_err(|_| Error::Checkpoint("invalid utf-8 in string".into()))?;
                        out.push_str(s);
                        self.pos += len - 1;
                    }
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        ensure!(self.pos + 4 <= self.bytes.len(), Checkpoint, "truncated \\u escape");
        let mut v = 0u32;
        for i in 0..4 {
            let c = self.bytes[self.pos + i];
            let d = (c as char).to_digit(16).ok_or_else(|| {
                Error::Checkpoint(format!("invalid hex digit '{}' in \\u escape", c as char))
            })?;
            v = (v << 4) | d;
        }
        self.pos += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self
            .peek()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| Error::Checkpoint("invalid number".into()))?;
        if !is_float {
            if let Ok(v) = text.parse::<u64>() {
                return Ok(Json::U64(v));
            }
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Json::I64(v));
            }
        }
        text.parse::<f64>()
            .map(Json::F64)
            .map_err(|_| Error::Checkpoint(format!("invalid number '{text}'")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_basics() {
        let src = r#"{"a": [1, 2.5, -3], "b": "hi\nthere", "c": {"d": null, "e": true}, "f": 18446744073709551615}"#;
        let v = Json::parse(src).unwrap();
        assert_eq!(v.get("b").unwrap().as_str().unwrap(), "hi\nthere");
        assert_eq!(v.get("a").unwrap().as_arr().unwrap().len(), 3);
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[0], Json::U64(1));
        assert_eq!(v.get("a").unwrap().as_arr().unwrap()[1], Json::F64(2.5));
        assert_eq!(
            v.get("f").unwrap(),
            &Json::U64(u64::MAX)
        );
        let back = Json::parse(&v.to_json_string()).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn rejects_garbage() {
        assert!(Json::parse("{").is_err());
        assert!(Json::parse("[1,]").is_err());
        assert!(Json::parse("").is_err());
        assert!(Json::parse("tru").is_err());
        assert!(Json::parse("\"abc").is_err());
        assert!(Json::parse("{\"a\":1} trailing").is_err());
        assert!(Json::parse("\"\\u12\"").is_err());
    }

    #[test]
    fn depth_bounded() {
        let mut s = String::new();
        for _ in 0..200 {
            s.push('[');
        }
        assert!(matches!(
            Json::parse(&s),
            Err(Error::CapacityExceeded(_))
        ));
    }

    #[test]
    fn unicode_escapes() {
        let v = Json::parse(r#""é😀""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "é😀");
        let v = Json::parse("\"\\u00e9\\ud83d\\ude00\"").unwrap();
        assert_eq!(v.as_str().unwrap(), "é😀");
    }

    #[test]
    fn typed_accessors() {
        let v = Json::parse(r#"{"n": 5, "x": -2, "pi": 3.5}"#).unwrap();
        assert_eq!(v.u64_field("n"), Some(5));
        assert_eq!(v.get("x").unwrap().as_i64(), Some(-2));
        assert_eq!(v.f64_field("pi"), Some(3.5));
        assert_eq!(v.get("x").unwrap().as_u64(), None);
    }
}
