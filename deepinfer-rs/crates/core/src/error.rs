//! Error types shared across the DeepInfer-rs workspace.

use std::fmt;

/// The error type for all DeepInfer-rs crates.
///
/// All fallible operations return [`crate::Result`] = `Result<T, Error>`.
/// Errors are plain data (no backtraces, no panics) so they can cross
/// thread and API boundaries safely. Every variant carries a
/// human-readable message; where structured context is useful, callers
/// format it into the message at construction time.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// A tensor shape did not match the operation's expectation.
    DimensionMismatch(String),
    /// Shape/stride combination cannot be used together for this operation.
    ShapeIncompatible(String),
    /// A caller-supplied argument was invalid.
    InvalidArgument(String),
    /// A value was outside its permitted range.
    OutOfRange(String),
    /// The requested dtype is not supported by this operation.
    UnsupportedDType(String),
    /// I/O or filesystem failure.
    Io(String),
    /// Environment variable could not be applied.
    Env(String),
    /// Checkpoint (safetensors / config) loading failure.
    Checkpoint(String),
    /// Grammar / structured-output compilation failure.
    Grammar(String),
    /// A hard resource limit was hit (used to fail fast instead of
    /// allocating unbounded memory — a denial-of-service guard).
    CapacityExceeded(String),
    /// A numerical routine detected an invalid input (NaN, non-finite, ...).
    Numerical(String),
    /// Internal invariant violation (a bug).
    Internal(String),
}

impl Error {
    /// Shorthand for [`Error::InvalidArgument`].
    pub fn invalid(msg: impl Into<String>) -> Error {
        Error::InvalidArgument(msg.into())
    }

    /// Shorthand for [`Error::Internal`].
    pub fn internal(msg: impl Into<String>) -> Error {
        Error::Internal(msg.into())
    }

    /// Shorthand for [`Error::Checkpoint`].
    pub fn checkpoint(msg: impl Into<String>) -> Error {
        Error::Checkpoint(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch(m) => write!(f, "dimension mismatch: {m}"),
            Error::ShapeIncompatible(m) => write!(f, "incompatible shapes: {m}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::OutOfRange(m) => write!(f, "out of range: {m}"),
            Error::UnsupportedDType(m) => write!(f, "unsupported dtype: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::Env(m) => write!(f, "environment error: {m}"),
            Error::Checkpoint(m) => write!(f, "checkpoint error: {m}"),
            Error::Grammar(m) => write!(f, "grammar error: {m}"),
            Error::CapacityExceeded(m) => write!(f, "capacity exceeded: {m}"),
            Error::Numerical(m) => write!(f, "numerical error: {m}"),
            Error::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e.to_string())
    }
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Evaluates `$cond`; on failure returns
/// `Err(Error::$variant(format!($($fmt)*)))`.
///
/// `$variant` must be one of the tuple-style variants (all of them are).
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $variant:ident, $($fmt:tt)*) => {
        if !($cond) {
            return Err($crate::Error::$variant(format!($($fmt)*)));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_informative() {
        let e = Error::DimensionMismatch("matmul: expected [2, 3], got [3, 2]".into());
        assert!(e.to_string().contains("matmul"));
        assert!(e.to_string().contains("[2, 3]"));
    }

    #[test]
    fn io_from_std() {
        let e: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "gone").into();
        assert!(matches!(e, Error::Io(_)));
    }

    fn check(cond: bool) -> Result<()> {
        ensure!(cond, OutOfRange, "value {} is bad", 42);
        Ok(())
    }

    #[test]
    fn ensure_macro_shapes_errors() {
        assert!(check(true).is_ok());
        let e = check(false).unwrap_err();
        assert!(matches!(e, Error::OutOfRange(_)));
        assert!(e.to_string().contains("42"));
    }
}
