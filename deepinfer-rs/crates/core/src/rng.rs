//! Deterministic, platform-independent random number generation.
//!
//! [`Philox`] implements Philox4x32-10 from the Random123 family
//! (Salmon et al., SC 2011) — the same counter-based algorithm used by
//! CUDA/PyTorch. All arithmetic is wrapping `u32` math, so streams are
//! bit-identical on every platform, which is the foundation of
//! DeepInfer's *deterministic sampling* and *batch-invariant inference*:
/// a sequence's samples depend only on its own `(seed, stream)` pair,
/// never on batch composition or scheduling order.

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const PHILOX_ROUNDS: usize = 10;

#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let prod = (a as u64) * (b as u64);
    ((prod >> 32) as u32, prod as u32)
}

/// SplitMix64 — used only to derive initial states from user seeds.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Philox4x32-10 counter-based RNG with a 128-bit counter.
#[derive(Debug, Clone)]
pub struct Philox {
    key: [u32; 2],
    counter: [u32; 4],
    buffer: [u32; 4],
    buf_idx: usize,
}

impl Philox {
    /// Creates a generator. `seed` selects the key, `stream` selects the
    /// initial counter block — different streams never overlap for any
    /// practical draw count.
    pub fn new(seed: u64, stream: u64) -> Philox {
        let mut sm = SplitMix64::new(seed);
        let key = [sm.next_u64() as u32, (sm.next_u64() >> 32) as u32];
        let counter = [
            stream as u32,
            (stream >> 32) as u32,
            sm.next_u64() as u32,
            (sm.next_u64() >> 32) as u32,
        ];
        Philox {
            key,
            counter,
            buffer: [0; 4],
            buf_idx: 4,
        }
    }

    /// Advances the 128-bit counter by one block.
    fn bump_counter(&mut self) {
        for w in self.counter.iter_mut() {
            let (nw, overflow) = w.overflowing_add(1);
            *w = nw;
            if !overflow {
                return;
            }
        }
    }

    /// Generates one block of four `u32`s (the Philox4x32 round function).
    fn gen_block(&self, mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
        for r in 0..PHILOX_ROUNDS {
            let (hi0, lo0) = mulhilo32(PHILOX_M0, ctr[0]);
            let (hi1, lo1) = mulhilo32(PHILOX_M1, ctr[2]);
            ctr = [
                hi1 ^ ctr[1] ^ key[0],
                lo1,
                hi0 ^ ctr[3] ^ key[1],
                lo0,
            ];
            if r + 1 < PHILOX_ROUNDS {
                key[0] = key[0].wrapping_add(PHILOX_W0);
                key[1] = key[1].wrapping_add(PHILOX_W1);
            }
        }
        ctr
    }

    /// Next raw `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.buf_idx >= 4 {
            self.buffer = self.gen_block(self.counter, self.key);
            self.buf_idx = 0;
            self.bump_counter();
        }
        let v = self.buffer[self.buf_idx];
        self.buf_idx += 1;
        v
    }

    /// Next uniform `f32` in `[0, 1)` with 24 bits of mantissa entropy.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }

    /// Next uniform `f64` in `[0, 1)` with 53 bits of entropy.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() as u64) >> 5;
        let b = (self.next_u32() as u64) >> 6;
        ((a << 26) | b) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn uniform_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform `usize` in `[lo, hi)` (rejection-free, slight modulo bias
    /// documented: at most 2^-32 for ranges below 2^32).
    pub fn uniform_usize(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(hi > lo);
        let span = (hi - lo) as u64;
        lo + (self.next_u64_from_two() % span) as usize
    }

    /// Uniform `u64` built from two `u32` draws.
    pub fn next_u64_from_two(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// Standard-normal `f32` via Box–Muller.
    pub fn normal_f32(&mut self) -> f32 {
        let mut u1 = self.next_f32();
        if u1 <= f32::EPSILON {
            u1 = f32::EPSILON;
        }
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Fills `out` with uniform `f32` in `[lo, hi)`.
    pub fn fill_uniform(&mut self, out: &mut [f32], lo: f32, hi: f32) {
        for v in out.iter_mut() {
            *v = self.uniform_f32(lo, hi);
        }
    }

    /// Fills `out` with standard normals (Box–Muller, deterministic order).
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.normal_f32();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_instances() {
        let mut a = Philox::new(42, 0);
        let mut b = Philox::new(42, 0);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_streams_differ() {
        let mut a = Philox::new(42, 0);
        let mut b = Philox::new(42, 1);
        let mut diffs = 0;
        for _ in 0..64 {
            if a.next_u32() != b.next_u32() {
                diffs += 1;
            }
        }
        assert!(diffs > 32, "streams look identical: {diffs}/64 differ");
    }

    #[test]
    fn uniform_bounds() {
        let mut rng = Philox::new(7, 3);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
            let u = rng.uniform_f32(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&u));
        }
    }

    #[test]
    fn normal_has_variance() {
        let mut rng = Philox::new(11, 22);
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            let x = rng.normal_f32() as f64;
            sum += x;
            sum2 += x * x;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn uniform_usize_in_range() {
        let mut rng = Philox::new(5, 5);
        for _ in 0..1000 {
            let v = rng.uniform_usize(3, 9);
            assert!((3..9).contains(&v));
        }
    }
}
