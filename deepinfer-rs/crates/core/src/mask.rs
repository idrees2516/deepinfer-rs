//! Vocabulary-sized boolean masks for structured-output-constrained
//! decoding (produced by `deepinfer-grammar`, consumed by
//! `deepinfer-sampler` and the runtime).

/// A compact bitset over token ids `[0, vocab)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMask {
    bits: Vec<u64>,
    vocab: usize,
    any_set: bool,
}

impl TokenMask {
    /// All tokens allowed.
    pub fn all_ones(vocab: usize) -> TokenMask {
        let words = vocab.div_ceil(64);
        let mut bits = vec![u64::MAX; words];
        if vocab % 64 != 0 {
            let last = words - 1;
            bits[last] &= (1u64 << (vocab % 64)) - 1;
        }
        TokenMask {
            bits,
            vocab,
            any_set: vocab > 0,
        }
    }

    /// No tokens allowed.
    pub fn zeros(vocab: usize) -> TokenMask {
        TokenMask {
            bits: vec![0; vocab.div_ceil(64)],
            vocab,
            any_set: false,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Allows token `id`.
    pub fn set(&mut self, id: u32) {
        let id = id as usize;
        assert!(id < self.vocab, "token id {id} out of vocab {vocab}", vocab = self.vocab);
        self.bits[id / 64] |= 1u64 << (id % 64);
        self.any_set = true;
    }

    /// Disallows token `id`.
    pub fn clear(&mut self, id: u32) {
        let id = id as usize;
        assert!(id < self.vocab);
        self.bits[id / 64] &= !(1u64 << (id % 64));
    }

    /// `true` if token `id` is allowed (out-of-range ids are disallowed).
    pub fn test(&self, id: u32) -> bool {
        let id = id as usize;
        if id >= self.vocab {
            return false;
        }
        (self.bits[id / 64] >> (id % 64)) & 1 == 1
    }

    /// Intersects with `other` (both must cover the same vocab).
    pub fn intersect(&mut self, other: &TokenMask) {
        assert_eq!(self.vocab, other.vocab, "vocab mismatch");
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
        self.any_set = self.bits.iter().any(|w| *w != 0);
    }

    /// Number of allowed tokens.
    pub fn count_ones(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// `true` if at least one token is allowed.
    pub fn any(&self) -> bool {
        self.any_set
    }

    /// Iterates over allowed token ids in ascending order.
    pub fn iter_ones(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits
            .iter()
            .enumerate()
            .flat_map(|(wi, &word)| {
                (0..64).filter_map(move |b| {
                    if (word >> b) & 1 == 1 {
                        Some((wi * 64 + b) as u32)
                    } else {
                        None
                    }
                })
            })
            .take_while(|id| (*id as usize) < self.vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_test_clear() {
        let mut m = TokenMask::zeros(200);
        assert!(!m.any());
        m.set(0);
        m.set(63);
        m.set(64);
        m.set(199);
        assert!(m.test(0) && m.test(63) && m.test(64) && m.test(199));
        assert!(!m.test(65));
        assert_eq!(m.count_ones(), 4);
        m.clear(63);
        assert!(!m.test(63));
        assert!(!m.test(200));
        assert!(!m.test(u32::MAX));
    }

    #[test]
    fn all_ones_partial_word() {
        let m = TokenMask::all_ones(70);
        assert_eq!(m.count_ones(), 70);
        assert!(m.test(69));
        assert!(!m.test(70));
        let ones: Vec<u32> = m.iter_ones().collect();
        assert_eq!(ones.len(), 70);
        assert_eq!(ones[69], 69);
    }

    #[test]
    fn intersection() {
        let mut a = TokenMask::all_ones(128);
        let mut b = TokenMask::zeros(128);
        b.set(5);
        b.set(100);
        a.intersect(&b);
        assert_eq!(a.count_ones(), 2);
        assert!(a.test(5) && a.test(100));
    }
}
