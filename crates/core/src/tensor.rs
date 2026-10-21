//! Dense, immutable, dtype-tagged tensors.

use crate::dtype::DType;
use crate::ensure;
use crate::error::{Error, Result};
use crate::format::{
    bf16_to_f32, e2m1_to_f32, e4m3_to_f32, e5m2_to_f32, e8m0_to_f32, f16_to_f32,
};
use std::sync::Arc;

/// A dense, contiguous, immutable tensor backed by reference-counted bytes.
///
/// Tensors are plain data: cheap to clone (the buffer is shared through an
/// `Arc`), safe to send across threads, and always validated on
/// construction so that every later access can rely on shape/size
/// invariants.
#[derive(Debug, Clone)]
pub struct Tensor {
    dtype: DType,
    shape: Vec<usize>,
    data: Arc<Vec<u8>>,
}

impl Tensor {
    /// Builds a tensor from raw bytes, validating that the buffer size
    /// matches `shape` × `dtype` size exactly.
    pub fn from_raw(dtype: DType, shape: Vec<usize>, data: Vec<u8>) -> Result<Tensor> {
        // Product of an empty shape is 1 (scalar tensor semantics).
        let numel: usize = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| Error::CapacityExceeded("tensor numel overflow".into()))?;
        let expected = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::CapacityExceeded("tensor byte size overflow".into()))?;
        if data.len() != expected {
            return Err(Error::DimensionMismatch(format!(
                "raw buffer for {dtype} tensor of shape {shape:?}: {expected} bytes expected, {} given",
                data.len()
            )));
        }
        Ok(Tensor {
            dtype,
            shape,
            data: Arc::new(data),
        })
    }

    /// Builds an `f32` tensor from a flat row-major buffer.
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor> {
        let numel: usize = shape.iter().copied().product();
        if data.len() != numel {
            return Err(Error::DimensionMismatch(format!(
                "from_f32: shape {shape:?} needs {} elems, got {}",
                numel,
                data.len()
            )));
        }
        let mut bytes = Vec::with_capacity(numel * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Tensor::from_raw(DType::F32, shape, bytes)
    }

    /// Allocates a zero-filled tensor.
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Result<Tensor> {
        let numel: usize = shape.iter().copied().product();
        let len = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::CapacityExceeded("tensor byte size overflow".into()))?;
        Tensor::from_raw(dtype, shape, vec![0u8; len])
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape (row-major, contiguous).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Total number of bytes.
    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    /// Raw bytes (row-major, little-endian for multi-byte types).
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Materializes the tensor as flat `f32` (dequantizing quantized
    /// formats element-wise; `E8M0` becomes its power-of-two value).
    ///
    /// Returns an error for integer tensors larger than 2^24 elements per
    /// value being outside the exact f32 range is not checked — integer
    /// conversion is exact only for `|v| <= 2^24`; callers that need exact
    /// integers must use [`Tensor::bytes`] instead.
    pub fn as_f32_flat(&self) -> Result<Vec<f32>> {
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            DType::F32 => {
                for i in 0..n {
                    let b: [u8; 4] = self.data[i * 4..i * 4 + 4].try_into().unwrap();
                    out.push(f32::from_le_bytes(b));
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    let b: [u8; 2] = self.data[i * 2..i * 2 + 2].try_into().unwrap();
                    out.push(bf16_to_f32(u16::from_le_bytes(b)));
                }
            }
            DType::F16 => {
                for i in 0..n {
                    let b: [u8; 2] = self.data[i * 2..i * 2 + 2].try_into().unwrap();
                    out.push(f16_to_f32(u16::from_le_bytes(b)));
                }
            }
            DType::E4M3 => {
                for i in 0..n {
                    out.push(e4m3_to_f32(self.data[i]));
                }
            }
            DType::E5M2 => {
                for i in 0..n {
                    out.push(e5m2_to_f32(self.data[i]));
                }
            }
            DType::E8M0 => {
                for i in 0..n {
                    out.push(e8m0_to_f32(self.data[i]));
                }
            }
            DType::E2M1 => {
                for i in 0..n {
                    out.push(e2m1_to_f32(self.data[i]));
                }
            }
            other => {
                return Err(Error::UnsupportedDType(format!(
                    "as_f32_flat not meaningful for {other}"
                )))
            }
        }
        Ok(out)
    }

    /// Reads one element as `f32` (dequantized).
    pub fn get_f32(&self, index: usize) -> Result<f32> {
        ensure!(index < self.numel(), OutOfRange, "index {index} >= numel {}", self.numel());
        let i = index;
        Ok(match self.dtype {
            DType::F32 => {
                let b: [u8; 4] = self.data[i * 4..i * 4 + 4].try_into().unwrap();
                f32::from_le_bytes(b)
            }
            DType::BF16 => {
                let b: [u8; 2] = self.data[i * 2..i * 2 + 2].try_into().unwrap();
                bf16_to_f32(u16::from_le_bytes(b))
            }
            DType::F16 => {
                let b: [u8; 2] = self.data[i * 2..i * 2 + 2].try_into().unwrap();
                f16_to_f32(u16::from_le_bytes(b))
            }
            DType::E4M3 => e4m3_to_f32(self.data[i]),
            DType::E5M2 => e5m2_to_f32(self.data[i]),
            DType::E8M0 => e8m0_to_f32(self.data[i]),
            DType::E2M1 => e2m1_to_f32(self.data[i]),
            other => {
                return Err(Error::UnsupportedDType(format!(
                    "get_f32 not meaningful for {other}"
                )))
            }
        })
    }

    /// Returns a sub-tensor consisting of rows `[row_start, row_end)` of a
    /// 2-D tensor (`[rows, cols]`), sharing the underlying bytes where
    /// possible (contiguous copy for simplicity and safety).
    pub fn rows(&self, row_start: usize, row_end: usize) -> Result<Tensor> {
        ensure!(self.shape.len() >= 1, InvalidArgument, "rows() needs >= 1 dim");
        if self.shape.len() == 1 {
            // Treat a 1-D tensor as a column of rows of length 1.
            ensure!(row_end <= self.shape[0], OutOfRange, "row range");
            let mut out = Vec::with_capacity((row_end - row_start) * self.dtype.size_in_bytes());
            let s = self.dtype.size_in_bytes();
            out.extend_from_slice(&self.data[row_start * s..row_end * s]);
            return Tensor::from_raw(self.dtype, vec![row_end - row_start], out);
        }
        ensure!(self.shape.len() == 2, InvalidArgument, "rows() on >2-D tensors is not supported");
        let cols = self.shape[1];
        ensure!(row_end <= self.shape[0] && row_start <= row_end, OutOfRange, "row range {row_start}..{row_end} of {}", self.shape[0]);
        let s = self.dtype.size_in_bytes();
        let mut out =
            Vec::with_capacity((row_end.saturating_sub(row_start)) * cols * s);
        for r in row_start..row_end {
            let start = r * cols * s;
            out.extend_from_slice(&self.data[start..start + cols * s]);
        }
        Tensor::from_raw(self.dtype, vec![row_end - row_start, cols], out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trip() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.nbytes(), 16);
        assert_eq!(t.as_f32_flat().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.get_f32(3).unwrap(), 4.0);
    }

    #[test]
    fn rejects_bad_buffer() {
        assert!(Tensor::from_raw(DType::F32, vec![2, 2], vec![0; 15]).is_err());
        assert!(Tensor::from_raw(DType::F32, vec![2, 2], vec![0; 16]).is_ok());
    }

    #[test]
    fn bf16_tensor_dequantizes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::format::f32_to_bf16_rne(1.0).to_le_bytes());
        bytes.extend_from_slice(&crate::format::f32_to_bf16_rne(-2.5).to_le_bytes());
        let t = Tensor::from_raw(DType::BF16, vec![2], bytes).unwrap();
        let f = t.as_f32_flat().unwrap();
        assert!((f[0] - 1.0).abs() < 1e-6);
        assert!((f[1] + 2.5).abs() < 0.01);
    }

    #[test]
    fn rows_slice() {
        let t = Tensor::from_f32((0..12).map(|i| i as f32).collect(), vec![4, 3]).unwrap();
        let r = t.rows(1, 3).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.as_f32_flat().unwrap(), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(t.rows(3, 5).is_err());
    }

    #[test]
    fn get_f32_bounds_checked() {
        let t = Tensor::from_f32(vec![1.0], vec![1]).unwrap();
        assert!(t.get_f32(1).is_err());
        assert_eq!(t.get_f32(0).unwrap(), 1.0);
    }
}
