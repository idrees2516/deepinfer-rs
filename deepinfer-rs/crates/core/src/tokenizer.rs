//! Tokenization: a trait plus a deterministic byte-fallback tokenizer (with
//! optional special tokens) and a GPT-2-style byte-level BPE loader for
//! `tokenizer.json`.
//!
//! DeepInfer's reference runtime uses the HF tokenizer of the model card;
//! this CPU port provides equivalent coverage with zero external
//! dependencies. The BPE implementation covers the standard
//! `tokenizer.json` "model" section (vocab + merges) with the GPT-2
//! pre-tokenization pattern approximated by a hand-rolled scanner.
//!
//! The [`ByteTokenizer`] is the **universal fallback**: it tokenizes any
//! input, round-trips exactly, and — with [`ByteTokenizer::chat_specials`]
//! — carries the DeepSeek conversation control pieces as single tokens so
//! the chat template stays compact even without a `tokenizer.json`.
use crate::ensure;

use crate::error::{Error, Result};
use crate::json::Json;
use std::collections::HashMap;

/// A deterministic tokenizer.
pub trait Tokenizer: Send + Sync {
    /// Vocabulary size (ids in `[0, vocab_size)`).
    fn vocab_size(&self) -> usize;
    /// Encodes `text` into token ids.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    /// Decodes token ids back to text (lossy where the ids demand it).
    fn decode(&self, ids: &[u32]) -> Result<String>;
    /// The exact UTF-8 bytes of the single token `id` — unlike
    /// [`Tokenizer::decode`], never lossy and never a replacement
    /// character. Used to build the vocabulary byte table that
    /// structured-output masking consumes: a byte-level token holding
    /// `0xFF` must mask as the byte `0xFF`, not as U+FFFD.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] when `id` is outside the vocabulary.
    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let s = self.decode(&[id])?;
        Ok(s.into_bytes())
    }
    /// End-of-sequence token id.
    fn eos_token_id(&self) -> u32;
    /// Optional beginning-of-sequence id.
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
}

/// One special (control) piece of a [`ByteTokenizer`]: a literal string
/// with a dedicated id at or above 256.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpecialPiece {
    /// The piece as a string (must be non-empty UTF-8).
    piece: String,
    /// The piece's UTF-8 bytes (match key).
    bytes: Vec<u8>,
    /// The dedicated token id (`>= 256`).
    id: u32,
}

/// Byte-fallback tokenizer: 256 tokens, one per byte, plus optional
/// special control pieces with dedicated ids at or above 256. Infallible
/// and perfectly round-tripping — used as the default for tests, for
/// `--tiny` self-checks, and as the serving fallback when a checkpoint
/// ships no (or an unreadable) `tokenizer.json`.
///
/// With [`ByteTokenizer::chat_specials`] the DeepSeek conversation
/// pieces (`<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`,
/// `<｜User｜>`, `<｜Assistant｜>`) become single tokens (ids 256-259),
/// so the chat template costs four tokens instead of ~64 bytes — the
/// single-process serving default.
#[derive(Debug, Clone)]
pub struct ByteTokenizer {
    /// Special pieces sorted by descending byte length (greedy
    /// longest-match on encode).
    specials: Vec<SpecialPiece>,
    /// Special lookup by id (decode).
    by_id: HashMap<u32, String>,
    eos: u32,
    bos: Option<u32>,
}

impl ByteTokenizer {
    /// Creates a plain byte tokenizer (no special pieces); `eos` must
    /// be < 256.
    pub fn new(eos: u32) -> Result<ByteTokenizer> {
        ensure!(
            eos < 256,
            InvalidArgument,
            "byte tokenizer eos must be < 256, got {eos}"
        );
        Ok(ByteTokenizer {
            specials: Vec::new(),
            by_id: HashMap::new(),
            eos,
            bos: None,
        })
    }

    /// Creates a byte tokenizer with the given special pieces. Every
    /// id must be `>= 256` and unique; pieces must be non-empty; `eos`
    /// must either be a special id or `< 256`.
    pub fn with_specials(specials: &[(&str, u32)], eos: u32) -> Result<ByteTokenizer> {
        let mut pieces: Vec<SpecialPiece> = Vec::with_capacity(specials.len());
        let mut by_id = HashMap::with_capacity(specials.len());
        for (piece, id) in specials {
            ensure!(
                !piece.is_empty(),
                InvalidArgument,
                "special piece must be non-empty"
            );
            ensure!(
                *id >= 256,
                InvalidArgument,
                "special piece {piece:?} id {id} must be >= 256"
            );
            ensure!(
                !by_id.contains_key(id),
                InvalidArgument,
                "duplicate special id {id}"
            );
            ensure!(
                !pieces.iter().any(|p| p.bytes == piece.as_bytes()),
                InvalidArgument,
                "duplicate special piece {piece:?}"
            );
            by_id.insert(*id, (*piece).to_string());
            pieces.push(SpecialPiece {
                piece: (*piece).to_string(),
                bytes: piece.as_bytes().to_vec(),
                id: *id,
            });
        }
        // Greedy longest match: longest pieces first.
        pieces.sort_by(|a, b| b.bytes.len().cmp(&a.bytes.len()).then(a.id.cmp(&b.id)));
        let has_specials = !pieces.is_empty();
        ensure!(
            (eos as usize) < 256 || by_id.contains_key(&eos),
            InvalidArgument,
            "eos {eos} is neither a byte id (< 256) nor a declared special"
        );
        if !has_specials {
            // No specials: `by_id` is empty, so the check above reduced to
            // the plain-byte rule.
            ensure!(
                eos < 256,
                InvalidArgument,
                "byte tokenizer eos must be < 256, got {eos}"
            );
        }
        Ok(ByteTokenizer {
            specials: pieces,
            by_id,
            eos,
            bos: None,
        })
    }

    /// The DeepSeek conversation layout as special pieces (the pieces
    /// the server's chat template renders): BOS = 256, EOS = 257,
    /// `<｜User｜>` = 258, `<｜Assistant｜>` = 259. The returned
    /// tokenizer's `eos_token_id()` is 257 and `bos_token_id()` is 256.
    pub fn chat_specials() -> Result<ByteTokenizer> {
        let t = ByteTokenizer::with_specials(
            &[
                ("<｜begin▁of▁sentence｜>", 256),
                ("<｜end▁of▁sentence｜>", 257),
                ("<｜User｜>", 258),
                ("<｜Assistant｜>", 259),
            ],
            257,
        )?;
        Ok(ByteTokenizer {
            bos: Some(256),
            ..t
        })
    }

    /// The declared special pieces `(piece, id)` in id order.
    pub fn special_pieces(&self) -> Vec<(&str, u32)> {
        let mut v: Vec<(&str, u32)> = self
            .specials
            .iter()
            .map(|p| (p.piece.as_str(), p.id))
            .collect();
        v.sort_by_key(|(_, id)| *id);
        v
    }

    /// Looks up the special piece starting at byte offset `i` (greedy
    /// longest match); returns its id and byte length.
    fn match_special_at(&self, bytes: &[u8], i: usize) -> Option<(u32, usize)> {
        for p in &self.specials {
            let n = p.bytes.len();
            if bytes.len() >= i + n && &bytes[i..i + n] == p.bytes.as_slice() {
                return Some((p.id, n));
            }
        }
        None
    }
}

impl Tokenizer for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        256 + self.specials.len()
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let bytes = text.as_bytes();
        let mut ids = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if let Some((id, n)) = self.match_special_at(bytes, i) {
                ids.push(id);
                i += n;
            } else {
                ids.push(bytes[i] as u32);
                i += 1;
            }
        }
        Ok(ids)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            match self.by_id.get(&id) {
                Some(piece) => bytes.extend_from_slice(piece.as_bytes()),
                None => {
                    ensure!(
                        id < 256,
                        OutOfRange,
                        "byte tokenizer id {id} is not a byte or a special"
                    );
                    bytes.push(id as u8);
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        if let Some(piece) = self.by_id.get(&id) {
            return Ok(piece.as_bytes().to_vec());
        }
        ensure!(
            id < 256,
            OutOfRange,
            "byte tokenizer id {id} is not a byte or a special"
        );
        Ok(vec![id as u8])
    }
    fn eos_token_id(&self) -> u32 {
        self.eos
    }
    fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE
// ---------------------------------------------------------------------------

/// GPT-2 byte-to-unicode mapping (bytes → printable codepoints).
fn bytes_to_unicode() -> [char; 256] {
    let mut table = [char::REPLACEMENT_CHARACTER; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        if (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b) {
            table[b as usize] = char::from_u32(b).unwrap();
        } else {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

fn unicode_to_bytes() -> HashMap<char, u8> {
    let mut m = HashMap::with_capacity(256);
    for (b, c) in bytes_to_unicode().iter().enumerate() {
        m.insert(*c, b as u8);
    }
    m
}

/// Byte-level BPE tokenizer loaded from a parsed `tokenizer.json`.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
    u2b: HashMap<char, u8>,
    b2u: [char; 256],
    eos: u32,
    bos: Option<u32>,
}

impl BpeTokenizer {
    /// Builds from the `Json` value of a `tokenizer.json` file.
    pub fn from_tokenizer_json(json: &Json) -> Result<BpeTokenizer> {
        let model = json
            .get("model")
            .ok_or_else(|| Error::Checkpoint("tokenizer.json missing 'model'".into()))?;
        let vocab_obj = model
            .get("vocab")
            .ok_or_else(|| Error::Checkpoint("tokenizer.json missing 'model.vocab'".into()))?;
        let mut vocab = HashMap::new();
        for (token, id) in vocab_obj.entries() {
            let id = id
                .as_u64()
                .ok_or_else(|| Error::Checkpoint(format!("bad vocab id for {token:?}")))?;
            ensure!(
                id <= u32::MAX as u64,
                Checkpoint,
                "vocab id {id} exceeds u32"
            );
            vocab.insert(token.clone(), id as u32);
        }
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        let mut used = vec![false; max_id + 1];
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
            used[id as usize] = true;
        }
        ensure!(
            used.iter().all(|u| *u),
            Checkpoint,
            "tokenizer vocab has holes"
        );

        let mut merges = HashMap::new();
        if let Some(merges_arr) = model.get("merges").and_then(Json::as_arr) {
            for (rank, m) in merges_arr.iter().enumerate() {
                let s = m
                    .as_str()
                    .ok_or_else(|| Error::Checkpoint("non-string merge entry".into()))?;
                if let Some((a, b)) = s.split_once(' ') {
                    merges.insert((a.to_string(), b.to_string()), rank);
                }
            }
        }

        let eos = json
            .get("added_tokens")
            .and_then(Json::as_arr)
            .and_then(|arr| {
                arr.iter()
                    .find(|t| {
                        t.str_field("special") == Some("true")
                            && t.str_field("content") == Some("<|end_of_text|>")
                    })
                    .and_then(|t| t.u64_field("id"))
            })
            .map(|v| v as u32)
            .unwrap_or_else(|| vocab.get("<|end_of_text|>").copied().unwrap_or(0));
        let bos = json
            .get("added_tokens")
            .and_then(Json::as_arr)
            .and_then(|arr| {
                arr.iter()
                    .find(|t| {
                        t.str_field("special") == Some("true")
                            && t.str_field("content") == Some("<|begin_of_text|>")
                    })
                    .and_then(|t| t.u64_field("id"))
            })
            .map(|v| v as u32);

        Ok(BpeTokenizer {
            vocab,
            id_to_token,
            merges,
            u2b: unicode_to_bytes(),
            b2u: bytes_to_unicode(),
            eos,
            bos,
        })
    }

    /// GPT-2 pre-tokenization, approximated without a regex engine:
    /// `'x` contractions, runs of letters, runs of digits, runs of other
    /// non-space, trailing spaces handling — each chunk keeps its leading
    /// space when preceded by whitespace.
    fn pretokenize(text: &str) -> Vec<&str> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        let is_apostrophe_contraction = |b: &[u8], j: usize| -> bool {
            b.get(j) == Some(&b'\'')
                && matches!(
                    b.get(j + 1),
                    Some(b's') | Some(b't') | Some(b'm') | Some(b'd')
                )
                || (b.get(j) == Some(&b'\'')
                    && b.get(j + 1) == Some(&b'r')
                    && b.get(j + 2) == Some(&b'e'))
                || (b.get(j) == Some(&b'\'')
                    && b.get(j + 1) == Some(&b'v')
                    && b.get(j + 2) == Some(&b'e'))
                || (b.get(j) == Some(&b'\'')
                    && b.get(j + 1) == Some(&b'l')
                    && b.get(j + 2) == Some(&b'l'))
        };
        while i < bytes.len() {
            let start = i;
            // leading space attaches to the following word chunk
            if bytes[i] == b' ' && i + 1 < bytes.len() {
                i += 1;
            }
            let c = *bytes.get(i).unwrap_or(&b' ');
            if is_apostrophe_contraction(bytes, i) {
                i += if matches!(
                    bytes.get(i + 1),
                    Some(b's') | Some(b't') | Some(b'm') | Some(b'd')
                ) {
                    2
                } else {
                    3
                };
            } else if c.is_ascii_alphabetic() {
                while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
            } else if c.is_ascii_digit() {
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            } else if c.is_ascii_whitespace() {
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                // GPT-2: `\s+(?!\S)` keeps trailing whitespace runs except
                // the final space before a token; approximate by emitting
                // the run minus its last char when followed by non-space.
                if i < bytes.len() && i - start > 1 {
                    i -= 1;
                }
            } else {
                while i < bytes.len()
                    && !bytes[i].is_ascii_alphabetic()
                    && !bytes[i].is_ascii_digit()
                    && !bytes[i].is_ascii_whitespace()
                {
                    i += 1;
                }
            }
            if i == start {
                i += 1;
            }
            out.push(&text[start..i]);
        }
        out
    }

    fn bpe_merge(&self, word: &[String]) -> Vec<String> {
        if word.len() < 2 {
            return word.to_vec();
        }
        let mut pieces: Vec<String> = word.to_vec();
        loop {
            // Find the lowest-rank (earliest-index on ties) adjacent pair.
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..pieces.len() - 1 {
                if let Some(&rank) = self.merges.get(&(pieces[i].clone(), pieces[i + 1].clone())) {
                    match best {
                        Some((br, _)) if br <= rank => {}
                        _ => best = Some((rank, i)),
                    }
                }
            }
            let Some((_, idx)) = best else { break };
            let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
            pieces.splice(idx..idx + 2, std::iter::once(merged));
        }
        pieces
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn decode_token(&self, token: &str) -> Vec<u8> {
        token
            .chars()
            .filter_map(|c| self.u2b.get(&c).copied())
            .collect()
    }
}

impl Tokenizer for BpeTokenizer {
    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for chunk in Self::pretokenize(text) {
            let chars: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.b2u[b as usize].to_string())
                .collect();
            for piece in self.bpe_merge(&chars) {
                match self.token_to_id(&piece) {
                    Some(id) => ids.push(id),
                    None => {
                        // Unknown merged piece: fall back to per-char tokens
                        // then raw bytes.
                        for c in piece.chars() {
                            match self.token_to_id(&c.to_string()) {
                                Some(id) => ids.push(id),
                                None => {
                                    let bytes = self.decode_token(&c.to_string());
                                    if bytes.is_empty() {
                                        return Err(Error::InvalidArgument(format!(
                                            "cannot tokenize character {c:?}"
                                        )));
                                    }
                                    ids.push(bytes[0] as u32);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            ensure!(
                (id as usize) < self.id_to_token.len(),
                OutOfRange,
                "token id {id} >= vocab {}",
                self.id_to_token.len()
            );
            let tok = &self.id_to_token[id as usize];
            bytes.extend_from_slice(&self.decode_token(tok));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn eos_token_id(&self) -> u32 {
        self.eos
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trip() {
        let t = ByteTokenizer::new(0).unwrap();
        let s = "Hello, 世界! 🌊";
        let ids = t.encode(s).unwrap();
        assert_eq!(t.decode(&ids).unwrap(), s);
        assert_eq!(t.vocab_size(), 256);
        assert!(t.decode(&[256]).is_err());
        // Exact single-token bytes: a lone non-UTF-8 byte must not become
        // U+FFFD.
        assert_eq!(t.decode_token_bytes(0xFF).unwrap(), vec![0xFF]);
        assert_eq!(t.decode_token_bytes(b'a' as u32).unwrap(), b"a".to_vec());
        assert!(t.decode_token_bytes(256).is_err());
    }

    #[test]
    fn chat_specials_layout() {
        let t = ByteTokenizer::chat_specials().unwrap();
        assert_eq!(t.vocab_size(), 260);
        assert_eq!(t.eos_token_id(), 257);
        assert_eq!(t.bos_token_id(), Some(256));
        // The template renders to exactly four special tokens plus text
        // bytes — the piece itself never splits into byte tokens.
        let text = "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi<｜Assistant｜>";
        let ids = t.encode(text).unwrap();
        assert_eq!(ids[0], 256);
        assert_eq!(ids[1], b'Y' as u32);
        let user = ids.iter().position(|&i| i == 258).unwrap();
        assert_eq!(ids[user + 1], b'H' as u32);
        assert_eq!(*ids.last().unwrap(), 259);
        // Special pieces cost one token each: 1 (BOS) + 16 ("You are
        // helpful.") + 1 (User) + 2 ("Hi") + 1 (Assistant) = 21, vs 76
        // bytes of raw text (the pieces use 3-byte full-width chars).
        assert_eq!(text.len(), 76);
        assert_eq!(ids.len(), 21);
        // Round trip is exact (including the multi-byte pieces).
        assert_eq!(t.decode(&ids).unwrap(), text);
        // Token bytes of a special are its exact UTF-8.
        assert_eq!(
            t.decode_token_bytes(257).unwrap(),
            "<｜end▁of▁sentence｜>".as_bytes()
        );
        assert!(t.decode(&[260]).is_err());
        assert!(t.decode_token_bytes(260).is_err());
    }

    #[test]
    fn specials_longest_match_and_validation() {
        // Longest piece wins at a shared prefix.
        let t = ByteTokenizer::with_specials(&[("<a>", 300), ("<ab>", 301)], 300).unwrap();
        assert_eq!(t.encode("<ab>").unwrap(), vec![301]);
        assert_eq!(t.encode("<a>").unwrap(), vec![300]);
        assert_eq!(t.encode("<x").unwrap(), vec![b'<' as u32, b'x' as u32]);
        assert_eq!(t.decode(&[301]).unwrap(), "<ab>");
        assert_eq!(t.special_pieces(), vec![("<a>", 300), ("<ab>", 301)]);
        // Holes in the id space decode as errors.
        assert!(t.decode(&[299]).is_err());
        // Validation: id below 256, duplicate id, duplicate piece, bad eos.
        assert!(ByteTokenizer::with_specials(&[("<a>", 200)], 200).is_err());
        assert!(ByteTokenizer::with_specials(&[("<a>", 300), ("<b>", 300)], 300).is_err());
        assert!(ByteTokenizer::with_specials(&[("<a>", 300), ("<a>", 301)], 300).is_err());
        assert!(ByteTokenizer::with_specials(&[("<a>", 300)], 299).is_err());
        assert!(ByteTokenizer::with_specials(&[("", 300)], 300).is_err());
    }

    #[test]
    fn byte_unicode_table_bijective() {
        let u2b = unicode_to_bytes();
        let b2u = bytes_to_unicode();
        assert_eq!(u2b.len(), 256);
        for (i, c) in b2u.iter().enumerate() {
            assert_eq!(u2b[c], i as u8);
        }
    }

    #[test]
    fn bpe_from_json_round_trip() {
        // Minimal GPT-2-like tokenizer.json.
        let src = r#"{
            "model": {
                "vocab": {"l":0,"o":1,"w":2,"er":3,"low":4,"hello":5," ":6,"<|end_of_text|>":7},
                "merges": ["l o", "lo w", "lo w er"]
            },
            "added_tokens": [
                {"id":7,"content":"<|end_of_text|>","special":true}
            ]
        }"#;
        let j = crate::json::Json::parse(src).unwrap();
        let t = BpeTokenizer::from_tokenizer_json(&j).unwrap();
        assert_eq!(t.eos_token_id(), 7);
        // "low" merges: l o -> lo; lo w -> low.
        let ids = t.encode("low").unwrap();
        assert_eq!(ids, vec![4]);
        let ids = t.encode("lo").unwrap();
        assert_eq!(ids, vec![0, 1]);
        let back = t.decode(&[4]).unwrap();
        assert_eq!(back, "low");
    }

    #[test]
    fn pretokenize_chunks() {
        let chunks = BpeTokenizer::pretokenize("Hello world, don't 42!");
        assert!(chunks.contains(&"Hello"));
        assert!(chunks.contains(&" world"));
        assert!(chunks.contains(&","));
        assert!(chunks.contains(&" don"));
        assert!(chunks.contains(&"'t"));
        assert!(chunks.contains(&" 42"));
        assert!(chunks.contains(&"!"));
    }
}
