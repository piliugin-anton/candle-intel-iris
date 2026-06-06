use std::sync::{PoisonError, TryLockError};

/// wgpu-related errors.
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("no Intel GPU adapter found (vendor 0x8086)")]
    NoIntelAdapter,
    #[error("no Intel GPU adapter found with device id 0x{0:04x}")]
    IntelDeviceNotFound(u32),
    #[error("wgpu buffer map failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    #[error("wgpu device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("{0}")]
    Message(String),
    #[error("lock poisoned: {0}")]
    LockPoisoned(String),
    #[error("lock would block")]
    LockWouldBlock,
    #[error("internal adapter index {index} out of range (len {len})")]
    AdapterIndexOutOfRange { index: usize, len: usize },
}

impl<T> From<PoisonError<T>> for WgpuError {
    fn from(err: PoisonError<T>) -> Self {
        Self::LockPoisoned(err.to_string())
    }
}

impl<T> From<TryLockError<T>> for WgpuError {
    fn from(err: TryLockError<T>) -> Self {
        match err {
            TryLockError::Poisoned(p) => Self::LockPoisoned(p.to_string()),
            TryLockError::WouldBlock => Self::LockWouldBlock,
        }
    }
}

pub type Result<T> = std::result::Result<T, WgpuError>;
