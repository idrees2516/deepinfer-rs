//! Safetensors checkpoint reading with full validation.
//!
//! Format: `u64 LE header-length | JSON header | raw tensor data`.
//! Every offset, dtype and shape is validated before any byte is handed
//! out — malformed or hostile checkpoints fail with an error, never a
//! panic, and never an out-of-bounds access.
use crate::ensure;


use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::json::Json;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// One validated tensor descriptor from the header.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name.
    pub name: String,
    /// Element type.
    pub dtype: DType,
    /// Shape (row-major).
    pub shape: Vec<usize>,
    /// Byte range within the data region.
    pub data_offsets: (usize, usize),
}

/// An open safetensors file.
pub struct SafeTensorsFile {
    path: PathBuf,
    data_start: usize,
    meta: Vec<TensorMeta>,
    by_name: HashMap<String, usize>,
}

/// Maximum accepted header size (defensive bound).
const MAX_HEADER_BYTES: usize = 128 * 1024 * 1024;

impl SafeTensorsFile {
    /// Opens and validates `path`.
    pub fn open(path: &Path) -> Result<SafeTensorsFile> {
        let bytes = fs::read(path)
            .map_err(|e| Error::Checkpoint(format!("reading {}: {e}", path.display())))?;
        Self::parse(bytes, path.to_path_buf())
    }

    /// Parses an in-memory safetensors buffer (same layout as on disk).
    pub fn parse(bytes: Vec<u8>, path: PathBuf) -> Result<SafeTensorsFile> {
        ensure!(
            bytes.len() >= 8,
            Checkpoint,
            "{}: file too small for safetensors header",
            path.display()
        );
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        ensure!(
            header_len <= MAX_HEADER_BYTES,
            CapacityExceeded,
            "safetensors header of {header_len} bytes exceeds limit {MAX_HEADER_BYTES}"
        );
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Checkpoint("header length overflow".into()))?;
        ensure!(
            bytes.len() >= data_start,
            Checkpoint,
            "{}: truncated file (need {data_start} bytes for header)",
            path.display()
        );
        let header_text = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|_| Error::Checkpoint("header is not valid UTF-8".into()))?;
        let header = Json::parse(header_text)?;

        let mut meta = Vec::new();
        let mut by_name = HashMap::new();
        let mut entries: Vec<(String, (usize, usize))> = Vec::new();
        for (key, val) in header.entries() {
            if key == "__metadata__" {
                continue;
            }
            let dtype_str = val
                .str_field("dtype")
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: missing dtype")))?;
            let dtype = DType::from_safetensors_str(dtype_str).ok_or_else(|| {
                Error::UnsupportedDType(format!("tensor {key:?}: unknown dtype {dtype_str:?}"))
            })?;
            let shape = val
                .get("shape")
                .and_then(Json::as_arr)
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: missing shape")))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .filter(|v| *v <= usize::MAX as u64)
                        .map(|v| v as usize)
                        .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: bad dim")))
                })
                .collect::<Result<Vec<usize>>>()?;
            let offsets = val
                .get("data_offsets")
                .and_then(Json::as_arr)
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: missing data_offsets")))?;
            ensure!(
                offsets.len() == 2,
                Checkpoint,
                "tensor {key:?}: data_offsets must have 2 entries"
            );
            let begin = offsets[0]
                .as_u64()
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: bad begin offset")))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: bad end offset")))?;
            let begin = usize::try_from(begin).map_err(|_| Error::Checkpoint("offset overflow".into()))?;
            let end = usize::try_from(end).map_err(|_| Error::Checkpoint("offset overflow".into()))?;
            ensure!(end >= begin, Checkpoint, "tensor {key:?}: end < begin");
            let numel: usize = shape.iter().product();
            let expected = numel
                .checked_mul(dtype.size_in_bytes())
                .ok_or_else(|| Error::Checkpoint(format!("tensor {key:?}: size overflow")))?;
            ensure!(
                end - begin == expected,
                Checkpoint,
                "tensor {key:?}: byte range {}..{} does not match shape {shape:?} of {dtype} ({expected} bytes)",
                begin,
                end
            );
            let data_len = (bytes.len() - data_start) as u64;
            ensure!(
                end as u64 <= data_len,
                Checkpoint,
                "tensor {key:?}: offsets {}..{} exceed data region of {data_len} bytes",
                begin,
                end
            );
            ensure!(
                by_name.insert(key.clone(), meta.len()).is_none(),
                Checkpoint,
                "duplicate tensor name {key:?}"
            );
            entries.push((key.clone(), (begin, end)));
            meta.push(TensorMeta {
                name: key.clone(),
                dtype,
                shape,
                data_offsets: (begin, end),
            });
        }

        // Non-overlap check (sorted sweep).
        let mut spans = entries.iter().map(|(_, o)| *o).collect::<Vec<_>>();
        spans.sort_unstable();
        for w in spans.windows(2) {
            ensure!(
                w[0].1 <= w[1].0,
                Checkpoint,
                "overlapping tensor data ranges detected"
            );
        }

        Ok(SafeTensorsFile {
            path,
            data_start,
            meta,
            by_name,
        })
    }

    /// Names of all tensors.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.meta.iter().map(|m| m.name.as_str())
    }

    /// Metadata for a tensor.
    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|&i| &self.meta[i])
    }

    /// Loads one tensor (data region slice is copied out).
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let m = self
            .meta(name)
            .ok_or_else(|| Error::Checkpoint(format!("tensor {name:?} not found in {}", self.path.display())))?;
        let file = fs::File::open(&self.path)
            .map_err(|e| Error::Io(format!("{}: {e}", self.path.display())))?;
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file;
        let start = self.data_start as u64 + m.data_offsets.0 as u64;
        let len = m.data_offsets.1 - m.data_offsets.0;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| Error::Io(format!("seek: {e}")))?;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)
            .map_err(|e| Error::Io(format!("read {name:?}: {e}")))?;
        Tensor::from_raw(m.dtype, m.shape.clone(), buf)
    }
}

// Field only exists to document the relationship between bytes and file;
// the parse consumes the buffer.
impl SafeTensorsFile {
    fn _unused(&self) {}
}

/// A directory of sharded safetensors (with or without
/// `model.safetensors.index.json`).
pub struct CheckpointDir {
    dir: PathBuf,
    files: Vec<SafeTensorsFile>,
    index: HashMap<String, usize>,
}

impl CheckpointDir {
    /// Opens every `*.safetensors` under `dir` (non-recursive) and builds a
    /// name → file map. Uses the shard index when present.
    pub fn open(dir: &Path) -> Result<CheckpointDir> {
        ensure!(
            dir.is_dir(),
            Checkpoint,
            "checkpoint dir {} does not exist",
            dir.display()
        );
        let mut files = Vec::new();
        let mut index = HashMap::new();

        // Preferred order: via index, else lexicographic.
        let mut names: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(dir).map_err(|e| Error::Io(e.to_string()))? {
            let entry = entry.map_err(|e| Error::Io(e.to_string()))?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                names.push(p);
            }
        }
        names.sort();
        ensure!(!names.is_empty(), Checkpoint, "no *.safetensors in {}", dir.display());

        for p in names {
            let f = SafeTensorsFile::open(&p)?;
            for name in f.names() {
                index.insert(name.to_string(), files.len());
            }
            files.push(f);
        }
        Ok(CheckpointDir {
            dir: dir.to_path_buf(),
            files,
            index,
        })
    }

    /// Directory that was opened.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// All tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Loads one tensor by name.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let fi = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Checkpoint(format!("tensor {name:?} not found in {}", self.dir.display())))?;
        self.files[fi].tensor(name)
    }

    /// Number of shards.
    pub fn shard_count(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_st_file(path: &Path, name: &str) {
        // header: one f32 tensor shape [2,2]
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}}}"#
        )
        .into_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&2.0f32.to_le_bytes());
        out.extend_from_slice(&3.0f32.to_le_bytes());
        out.extend_from_slice(&4.0f32.to_le_bytes());
        fs::write(path, out).unwrap();
    }

    #[test]
    fn loads_valid_file() {
        let dir = std::env::temp_dir().join("di_core_st_test");
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.safetensors");
        write_st_file(&p, "w");
        let f = SafeTensorsFile::open(&p).unwrap();
        assert_eq!(f.names().collect::<Vec<_>>(), vec!["w"]);
        let t = f.tensor("w").unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.as_f32_flat().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(f.tensor("missing").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_truncated_and_bad_offsets() {
        let dir = std::env::temp_dir().join("di_core_st_bad");
        fs::create_dir_all(&dir).unwrap();
        // Truncated.
        let p = dir.join("bad1.safetensors");
        fs::write(&p, b"1234").unwrap();
        assert!(SafeTensorsFile::open(&p).is_err());
        // Offsets out of range.
        let header = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,64]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header);
        out.extend_from_slice(&[0u8; 16]);
        let p2 = dir.join("bad2.safetensors");
        fs::write(&p2, out).unwrap();
        assert!(SafeTensorsFile::open(&p2).is_err());
        // Shape/byte mismatch.
        let header = br#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header);
        out.extend_from_slice(&[0u8; 16]);
        let p3 = dir.join("bad3.safetensors");
        fs::write(&p3, out).unwrap();
        assert!(SafeTensorsFile::open(&p3).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_dir_aggregates() {
        let dir = std::env::temp_dir().join("di_core_st_dir");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_st_file(&dir.join("shard0.safetensors"), "w0");
        write_st_file(&dir.join("shard1.safetensors"), "w1");
        let d = CheckpointDir::open(&dir).unwrap();
        assert_eq!(d.shard_count(), 2);
        assert_eq!(d.names().count(), 2);
        assert!(d.tensor("w0").is_ok() && d.tensor("w1").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }
}
