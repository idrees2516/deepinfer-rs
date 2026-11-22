//! Chat-layer pieces: OpenAI-style messages, the DeepSeek chat
//! template, and the streaming UTF-8 delta decoder.
//!
//! The template follows the official DeepSeek-V3/V4 conversation
//! format (the one shipped in `tokenizer_config.json` of the model
//! cards, minus its Jinja scaffolding):
//!
//! ```text
//! <｜begin▁of▁sentence｜>{system}
//! <｜User｜>{user message}
//! <｜Assistant｜>{assistant message}<｜end▁of▁sentence｜>
//! <｜Assistant｜>          <- generation prompt
//! ```
//!
//! System messages concatenate directly after the BOS (joined by
//! newlines when several appear, matching the reference template's
//! `{%- set ns = ... %}` join), and the generation prompt appends
//! `<｜Assistant｜>` so the model continues as the assistant.

use deepinfer_core::{ensure, Result};

/// BOS piece of the DeepSeek tokenizer.
pub const BOS_PIECE: &str = "<｜begin▁of▁sentence｜>";
/// EOS piece of the DeepSeek tokenizer.
pub const EOS_PIECE: &str = "<｜end▁of▁sentence｜>";
/// User turn opener.
pub const USER_PIECE: &str = "<｜User｜>";
/// Assistant turn opener.
pub const ASSISTANT_PIECE: &str = "<｜Assistant｜>";

/// One chat message (the OpenAI `messages` entry subset we serve).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// `system` | `user` | `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// A validated chat request body (streaming flag and sampling knobs are
/// held by the caller; this is just the conversation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    /// The messages, in order.
    pub messages: Vec<ChatMessage>,
}

impl Conversation {
    /// Basic invariants: non-empty, known roles, non-empty content.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] describing the first violation.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.messages.is_empty(),
            InvalidArgument,
            "messages must contain at least one message"
        );
        for (i, m) in self.messages.iter().enumerate() {
            ensure!(
                matches!(m.role.as_str(), "system" | "user" | "assistant"),
                InvalidArgument,
                "messages[{i}]: unsupported role {:?} (system|user|assistant)",
                m.role
            );
            ensure!(
                !m.content.is_empty(),
                InvalidArgument,
                "messages[{i}]: content must be non-empty"
            );
        }
        Ok(())
    }

    /// Whether the last message is from the user (the natural
    /// continuation point).
    pub fn ends_with_user(&self) -> bool {
        self.messages
            .last()
            .map(|m| m.role == "user")
            .unwrap_or(false)
    }
}

/// Renders the DeepSeek chat template for `messages`.
///
/// `add_generation_prompt` appends the assistant opener so generation
/// continues the conversation as the assistant (OpenAI semantics: always
/// true for completion requests).
pub fn render_chat_template(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut out = String::with_capacity(128);
    out.push_str(BOS_PIECE);
    let mut system_parts: Vec<&str> = Vec::new();
    for m in messages {
        if m.role == "system" {
            system_parts.push(&m.content);
        }
    }
    out.push_str(&system_parts.join("\n"));
    for m in messages {
        match m.role.as_str() {
            "user" => {
                out.push_str(USER_PIECE);
                out.push_str(&m.content);
            }
            "assistant" => {
                out.push_str(ASSISTANT_PIECE);
                out.push_str(&m.content);
                out.push_str(EOS_PIECE);
            }
            _ => {}
        }
    }
    if add_generation_prompt {
        out.push_str(ASSISTANT_PIECE);
    }
    out
}

/// Buffers token bytes and flushes only complete UTF-8 sequences, so a
/// multi-byte character split across streamed tokens is emitted whole.
///
/// This is the standard "detokenize as you go" trick (vLLM/SGLang do
/// the same with `bytefy`): the model emits UTF-8 pieces; JSON
/// responses and SSE deltas must carry valid strings, so a partial
/// trailing sequence is held back until its continuation arrives (or
/// replaced with U+FFFD at end of stream).
#[derive(Debug, Default)]
pub struct DeltaDecoder {
    pending: Vec<u8>,
}

impl DeltaDecoder {
    /// A fresh decoder.
    pub fn new() -> DeltaDecoder {
        DeltaDecoder::default()
    }

    /// Feeds one token's bytes; returns the text that became
    /// renderable (possibly empty).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let complete = utf8_complete_prefix_len(&self.pending);
        let out = String::from_utf8_lossy(&self.pending[..complete]).into_owned();
        self.pending.drain(..complete);
        out
    }

    /// Flushes any trailing partial sequence as U+FFFD (end of stream).
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Length of the longest prefix of `bytes` that is complete UTF-8.
fn utf8_complete_prefix_len(bytes: &[u8]) -> usize {
    let mut i = 0;
    while i < bytes.len() {
        let need = match bytes[i] {
            0x00..=0x7F => 1,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return i, // invalid lead byte: stop (lossy handles it later)
        };
        let end = i + need;
        if end > bytes.len() {
            return i; // truncated sequence
        }
        // Validate continuation bytes.
        for &b in &bytes[i + 1..end] {
            if b & 0xC0 != 0x80 {
                return i;
            }
        }
        i = end;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn template_matches_deepseek_layout() {
        let convo = vec![
            msg("system", "You are helpful."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "Bye"),
        ];
        let s = render_chat_template(&convo, true);
        assert!(s.starts_with(BOS_PIECE));
        assert!(s.starts_with("<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi"));
        assert!(s.contains("<｜Assistant｜>Hello!<｜end▁of▁sentence｜>"));
        assert!(s.ends_with("<｜User｜>Bye<｜Assistant｜>"));
        // Without generation prompt the assistant opener is not appended.
        let s2 = render_chat_template(&convo, false);
        assert!(s2.ends_with("<｜User｜>Bye"));
    }

    #[test]
    fn conversation_validation() {
        let c = Conversation {
            messages: vec![msg("user", "hi")],
        };
        c.validate().unwrap();
        assert!(c.ends_with_user());
        let bad = Conversation {
            messages: vec![msg("tool", "x")],
        };
        assert!(bad.validate().is_err());
        let empty = Conversation { messages: vec![] };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn delta_decoder_holds_partial_utf8() {
        let mut d = DeltaDecoder::new();
        assert_eq!(d.push(b"ab"), "ab");
        // 中 = E4 B8 AD, split across three tokens.
        assert_eq!(d.push(&[0xE4]), "");
        assert_eq!(d.push(&[0xB8]), "");
        assert_eq!(d.push(&[0xAD]), "中");
        // Truncated at end of stream -> replacement character.
        assert_eq!(d.push(&[0xE4, 0xB8]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }

    #[test]
    fn utf8_prefix_lengths() {
        assert_eq!(utf8_complete_prefix_len(b"abc"), 3);
        assert_eq!(utf8_complete_prefix_len(&[0xE4, 0xB8]), 0);
        assert_eq!(utf8_complete_prefix_len(&[0xE4, 0xB8, 0xAD]), 3);
        assert_eq!(utf8_complete_prefix_len(&[b'a', 0xE4, 0xB8]), 1);
    }
}
