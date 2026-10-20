//! Paged KV cache with FlashMLA physical-index conventions.
//!
//! Ground truth: FlashMLA README — pages hold `[page_size,
//! bytes_per_token]` contiguously (page_size = 64 tokens); per-sequence
//! logical order is the block table (list of page ids); the physical
//! index of a token is `page_id * page_size + offset_in_page`
//! (`indices_in_kvcache` convention). Row formats: BF16 (2 B/dim), V4.1
//! FP8 528 B and FP4 288 B per 512-dim row (see [`crate::fp8cache`]).

use crate::config::CacheFormat;
use crate::fp8cache;
use deepinfer_core::{ensure, Error, Result};
use std::collections::BTreeMap;

/// Upper bound on one page's byte size (DoS guard for caller-chosen page
/// sizes).
const MAX_PAGE_BYTES: usize = 1 << 28;

/// Per-sequence cache state.
struct SeqEntry {
    /// Page ids in logical order (the block table).
    pages: Vec<u32>,
    /// Tokens stored for this sequence.
    tokens: usize,
}

/// Paged KV cache. Pages hold `[page_size, bytes_per_token]` for the
/// configured format; per-sequence page lists give logical order.
///
/// The row width (`head_dim`) is fixed by the first appended row and must
/// stay consistent afterwards. Pages are allocated lazily with
/// deterministic first-fit (lowest id) order; allocation fails with
/// [`Error::CapacityExceeded`] once `max_pages` is exhausted.
pub struct MlaPagedCache {
    format: CacheFormat,
    page_size: usize,
    max_pages: usize,
    pages: Vec<Vec<u8>>,
    /// Tokens written into each page (rows are appended sequentially, so
    /// a page's valid rows are the prefix `[0, page_fill)`).
    page_fill: Vec<u32>,
    /// Free-list of recycled page ids, kept descending so `pop()` is
    /// first-fit (lowest id). Pages are never freed through the public
    /// API; the list exists to keep allocation policy explicit.
    free: Vec<u32>,
    seqs: BTreeMap<u32, SeqEntry>,
    head_dim: Option<usize>,
}

impl MlaPagedCache {
    /// Creates an empty cache. `page_size` is the FlashMLA page size (64
    /// upstream; clamped to at least 1 so index arithmetic stays valid).
    pub fn new(format: CacheFormat, page_size: usize, max_pages: usize) -> Self {
        MlaPagedCache {
            format,
            page_size: page_size.max(1),
            max_pages,
            pages: Vec::new(),
            page_fill: Vec::new(),
            free: Vec::new(),
            seqs: BTreeMap::new(),
            head_dim: None,
        }
    }

    /// The configured row format.
    pub fn format(&self) -> CacheFormat {
        self.format
    }

    /// Tokens per page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// The row width fixed by the first append, or `None` while empty.
    pub(crate) fn head_dim(&self) -> Option<usize> {
        self.head_dim
    }

    fn bytes_per_token(&self) -> Option<usize> {
        fp8cache::row_bytes(self.format, self.head_dim?)
    }

    /// Allocates one page (first-fit over the free list, then a fresh id).
    fn alloc_page(&mut self) -> Result<u32> {
        if let Some(id) = self.free.pop() {
            return Ok(id);
        }
        ensure!(
            self.pages.len() < self.max_pages,
            CapacityExceeded,
            "paged KV cache exhausted: {} of {} pages allocated",
            self.pages.len(),
            self.max_pages
        );
        // Physical indices are i32: keep the page-id space representable.
        ensure!(
            (self.pages.len() as u64) * (self.page_size as u64) < i32::MAX as u64,
            CapacityExceeded,
            "page-id space exceeds the i32 physical-index range"
        );
        let bpt = self
            .bytes_per_token()
            .ok_or_else(|| Error::internal("bytes_per_token requires head_dim"))?;
        let len = self.page_size.checked_mul(bpt).ok_or_else(|| {
            Error::CapacityExceeded(format!(
                "page byte size overflows: {} x {bpt}",
                self.page_size
            ))
        })?;
        ensure!(
            len <= MAX_PAGE_BYTES,
            CapacityExceeded,
            "page byte size {len} exceeds the {MAX_PAGE_BYTES}-byte limit"
        );
        let mut page = Vec::new();
        page.try_reserve_exact(len).map_err(|_| {
            Error::CapacityExceeded(format!("cannot allocate a {len}-byte page"))
        })?;
        page.resize(len, 0u8);
        self.pages.push(page);
        self.page_fill.push(0);
        Ok((self.pages.len() - 1) as u32)
    }

    /// Appends one token row (`[head_dim]` f32, RoPE already applied by
    /// the caller) for `seq`, allocating pages as needed. The first append
    /// fixes the row width; the row must be finite and its length must be
    /// compatible with the format (FP8 requires `dim % 32 == 0`, FP4
    /// requires `dim % 16 == 0`).
    ///
    /// # Errors
    /// [`Error::CapacityExceeded`] when `max_pages` is exhausted;
    /// [`Error::DimensionMismatch`] / [`Error::InvalidArgument`] /
    /// [`Error::Numerical`] on inconsistent or non-finite rows.
    pub fn append(&mut self, seq: u32, row: &[f32]) -> Result<()> {
        ensure!(!row.is_empty(), InvalidArgument, "KV row must be non-empty");
        ensure!(
            row.iter().all(|v| v.is_finite()),
            Numerical,
            "KV row contains non-finite values"
        );
        match self.head_dim {
            None => {
                match self.format {
                    CacheFormat::Bf16 => {}
                    CacheFormat::Fp8Group32 => ensure!(
                        row.len() % fp8cache::FP8_GROUP == 0,
                        InvalidArgument,
                        "Fp8Group32 rows need a width divisible by {}, got {}",
                        fp8cache::FP8_GROUP,
                        row.len()
                    ),
                    CacheFormat::Fp4Group16 => ensure!(
                        row.len() % fp8cache::FP4_GROUP == 0,
                        InvalidArgument,
                        "Fp4Group16 rows need a width divisible by {}, got {}",
                        fp8cache::FP4_GROUP,
                        row.len()
                    ),
                }
                self.head_dim = Some(row.len());
            }
            Some(d) => ensure!(
                row.len() == d,
                DimensionMismatch,
                "KV row width {} does not match cache head_dim {d}",
                row.len()
            ),
        }
        let (mut page_ids, tokens) = match self.seqs.get(&seq) {
            Some(e) => (e.pages.clone(), e.tokens),
            None => (Vec::new(), 0),
        };
        let new_tokens = tokens
            .checked_add(1)
            .ok_or_else(|| Error::CapacityExceeded("token count overflow".into()))?;
        let needed = new_tokens.div_ceil(self.page_size);
        let mut newly: Vec<u32> = Vec::new();
        while page_ids.len() < needed {
            let p = self.alloc_page()?;
            page_ids.push(p);
            newly.push(p);
        }
        let bpt = self
            .bytes_per_token()
            .ok_or_else(|| Error::internal("head_dim unset after append validation"))?;
        let mut buf = vec![0u8; bpt];
        fp8cache::quantize_row(self.format, row, &mut buf)?;
        let off = tokens % self.page_size;
        let slot = tokens / self.page_size;
        let pid = page_ids[slot];
        let start = off
            .checked_mul(bpt)
            .ok_or_else(|| Error::internal("page offset overflow"))?;
        let page = &mut self.pages[pid as usize];
        page[start..start + bpt].copy_from_slice(&buf);
        self.page_fill[pid as usize] = (off + 1) as u32;
        let entry = self
            .seqs
            .entry(seq)
            .or_insert_with(|| SeqEntry {
                pages: Vec::new(),
                tokens: 0,
            });
        if !newly.is_empty() {
            entry.pages.extend_from_slice(&newly);
        }
        entry.tokens = new_tokens;
        Ok(())
    }

    /// Number of tokens stored for `seq` (0 for unknown sequences).
    pub fn seq_len(&self, seq: u32) -> usize {
        self.seqs.get(&seq).map(|e| e.tokens).unwrap_or(0)
    }

    /// Block table (page ids in logical order); empty for unknown
    /// sequences.
    pub fn block_table(&self, seq: u32) -> &[u32] {
        self.seqs
            .get(&seq)
            .map(|e| e.pages.as_slice())
            .unwrap_or(&[])
    }

    /// Encodes a logical token position of `seq` into the FlashMLA
    /// physical index `page_id * page_size + offset`.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] when `pos >= seq_len(seq)` or the encoded
    /// index does not fit in `i32`.
    pub fn physical_index(&self, seq: u32, pos: usize) -> Result<i32> {
        let tokens = self.seqs.get(&seq).map(|e| e.tokens).unwrap_or(0);
        ensure!(
            pos < tokens,
            OutOfRange,
            "position {pos} out of range for sequence {seq} (len {tokens})"
        );
        let pid = self.seqs[&seq].pages[pos / self.page_size];
        let off = pos % self.page_size;
        let v = pid as u64 * self.page_size as u64 + off as u64;
        i32::try_from(v)
            .map_err(|_| Error::OutOfRange(format!("physical index {v} exceeds i32 range")))
    }

    /// Dequantizes one stored token row by physical index.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] when `phys` is negative, references an
    /// unallocated page or an unwritten row (or the cache is empty).
    pub fn row_f32(&self, phys: i32) -> Result<Vec<f32>> {
        let d = self
            .head_dim
            .ok_or_else(|| Error::invalid("KV cache is empty; no rows have been appended"))?;
        ensure!(phys >= 0, OutOfRange, "physical index must be non-negative, got {phys}");
        let p = phys as usize / self.page_size;
        let off = phys as usize % self.page_size;
        ensure!(
            p < self.pages.len(),
            OutOfRange,
            "physical index {phys} references unallocated page {p}"
        );
        ensure!(
            (off as u32) < self.page_fill[p],
            OutOfRange,
            "physical index {phys} references an unwritten row"
        );
        let bpt = self
            .bytes_per_token()
            .ok_or_else(|| Error::internal("bytes_per_token requires head_dim"))?;
        let start = off * bpt;
        let mut out = vec![0.0f32; d];
        fp8cache::dequantize_row(self.format, &self.pages[p][start..start + bpt], &mut out);
        Ok(out)
    }

    /// Total stored tokens across all sequences.
    pub fn total_tokens(&self) -> usize {
        self.seqs.values().map(|e| e.tokens).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::format::{bf16_to_f32, f32_to_bf16_rne};
    use deepinfer_core::prop::assert_close_slice;

    fn rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = deepinfer_core::rng::Philox::new(seed, 7);
        (0..n)
            .map(|_| (0..d).map(|_| rng.uniform_f32(-1.0, 1.0)).collect())
            .collect()
    }

    #[test]
    fn canonical_page_byte_sizes_per_format() {
        // A page holds 64 tokens: bytes = 64 * bytes_per_token.
        for (format, bytes) in [
            (CacheFormat::Bf16, 64 * 1024usize),
            (CacheFormat::Fp8Group32, 64 * 528),
            (CacheFormat::Fp4Group16, 64 * 288),
        ] {
            let mut cache = MlaPagedCache::new(format, 64, 4);
            cache.append(0, &vec![0.25; 512]).unwrap();
            assert_eq!(cache.pages[0].len(), bytes, "{format:?}");
            assert_eq!(cache.page_size(), 64);
            assert_eq!(cache.format(), format);
        }
    }

    #[test]
    fn append_len_block_table_physical_index_roundtrip() {
        let data = rows(70, 32, 11);
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 8);
        for (t, row) in data.iter().enumerate() {
            cache.append(3, row).unwrap();
            assert_eq!(cache.seq_len(3), t + 1);
        }
        assert_eq!(cache.seq_len(3), 70);
        assert_eq!(cache.seq_len(99), 0);
        assert_eq!(cache.block_table(3), &[0, 1]);
        assert!(cache.block_table(99).is_empty());
        assert_eq!(cache.total_tokens(), 70);
        for (pos, row) in data.iter().enumerate() {
            let phys = cache.physical_index(3, pos).unwrap();
            let expected = (pos / 64) as i32 * 64 + (pos % 64) as i32;
            assert_eq!(phys, expected);
            let back = cache.row_f32(phys).unwrap();
            let quantized: Vec<f32> = row.iter().map(|&v| bf16_to_f32(f32_to_bf16_rne(v))).collect();
            assert_close_slice(&back, &quantized, 1e-12, "bf16 roundtrip");
        }
        assert!(cache.physical_index(3, 70).is_err());
        assert!(cache.physical_index(99, 0).is_err());
    }

    #[test]
    fn capacity_exhaustion_errors() {
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 2);
        let row = vec![0.5f32; 32];
        for _ in 0..128 {
            cache.append(0, &row).unwrap();
        }
        let err = cache.append(0, &row).unwrap_err();
        assert!(matches!(err, Error::CapacityExceeded(_)), "{err}");
        // The 128 appended tokens are still intact.
        assert_eq!(cache.seq_len(0), 128);
        assert_eq!(cache.block_table(0).len(), 2);
        assert!(cache.row_f32(0).is_ok());
    }

    #[test]
    fn page_wraparound_three_pages() {
        let data = rows(150, 16, 21);
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 10);
        for row in &data {
            cache.append(7, row).unwrap();
        }
        assert_eq!(cache.block_table(7).len(), 3);
        assert_eq!(cache.seq_len(7), 150);
        // Boundaries between pages decode the right logical rows.
        for pos in [0usize, 63, 64, 65, 127, 128, 149] {
            let phys = cache.physical_index(7, pos).unwrap();
            assert_eq!(phys, (pos / 64) as i32 * 64 + (pos % 64) as i32);
            let back = cache.row_f32(phys).unwrap();
            let q: Vec<f32> = data[pos]
                .iter()
                .map(|&v| bf16_to_f32(f32_to_bf16_rne(v)))
                .collect();
            assert_close_slice(&back, &q, 1e-12, "wraparound row");
        }
        // Last page only partially filled: offset 21 is the last written.
        assert_eq!(cache.page_fill[2], 22);
        assert!(cache.row_f32(2 * 64 + 22).is_err());
        assert!(cache.row_f32(2 * 64 + 63).is_err());
    }

    #[test]
    fn row_width_and_format_validation() {
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 4);
        cache.append(0, &vec![0.1; 32]).unwrap();
        assert!(matches!(
            cache.append(0, &vec![0.1; 33]).unwrap_err(),
            Error::DimensionMismatch(_)
        ));
        assert!(cache.append(0, &[]).is_err());
        let mut bad = vec![0.1f32; 32];
        bad[5] = f32::NAN;
        assert!(matches!(
            cache.append(0, &bad).unwrap_err(),
            Error::Numerical(_)
        ));
        // FP8 needs width % 32 == 0; FP4 needs % 16 == 0.
        let mut fp8 = MlaPagedCache::new(CacheFormat::Fp8Group32, 64, 4);
        assert!(fp8.append(0, &vec![0.1; 33]).is_err());
        fp8.append(0, &vec![0.1; 32]).unwrap();
        let mut fp4 = MlaPagedCache::new(CacheFormat::Fp4Group16, 64, 4);
        assert!(fp4.append(0, &vec![0.1; 40]).is_err());
        fp4.append(0, &vec![0.1; 48]).unwrap();
    }

    #[test]
    fn row_f32_bounds_and_empty_cache() {
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 4);
        assert!(cache.row_f32(0).is_err()); // empty cache
        cache.append(0, &vec![0.25; 16]).unwrap();
        assert!(cache.row_f32(-1).is_err());
        assert!(cache.row_f32(64).is_err()); // unallocated page
        assert!(cache.row_f32(1).is_err()); // allocated page, unwritten row
        assert!(cache.row_f32(0).is_ok());
    }

    #[test]
    fn multi_sequence_totals_and_isolation() {
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 8);
        let a = vec![0.5f32; 16];
        let b = vec![-0.25f32; 16];
        for _ in 0..70 {
            cache.append(1, &a).unwrap();
        }
        for _ in 0..10 {
            cache.append(2, &b).unwrap();
        }
        assert_eq!(cache.total_tokens(), 80);
        assert_eq!(cache.seq_len(1), 70);
        assert_eq!(cache.seq_len(2), 10);
        assert_ne!(cache.block_table(1), cache.block_table(2));
        // Sequence 2's first row is on its own page (allocated after
        // sequence 1's two pages).
        assert_eq!(cache.block_table(2)[0], 2);
        let row = cache.row_f32(cache.physical_index(2, 0).unwrap()).unwrap();
        assert!(row.iter().all(|&v| v == bf16_to_f32(f32_to_bf16_rne(-0.25))));
    }

    #[test]
    fn quantized_cache_roundtrip_all_formats() {
        let data = rows(3, 64, 31);
        for (format, tol) in [
            (CacheFormat::Bf16, 0.02),
            (CacheFormat::Fp8Group32, 0.06),
            (CacheFormat::Fp4Group16, 1.6),
        ] {
            let mut cache = MlaPagedCache::new(format, 64, 4);
            for row in &data {
                cache.append(0, row).unwrap();
            }
            for (pos, row) in data.iter().enumerate() {
                let back = cache.row_f32(cache.physical_index(0, pos).unwrap()).unwrap();
                assert_close_slice(&back, row, tol, "{format:?} row {pos}");
            }
        }
    }
}
