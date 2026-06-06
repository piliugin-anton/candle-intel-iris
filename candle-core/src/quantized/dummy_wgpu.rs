#![allow(unused)]
use super::GgmlDType;
use crate::{Error, Result};

pub struct QWgpuStorage {
    dtype: GgmlDType,
}

impl QWgpuStorage {
    pub fn zeros(_: &(), _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &() {
        &()
    }

    pub fn dequantize(&self, _: usize) -> Result<()> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn quantize(&mut self, _: &()) -> Result<()> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn quantize_imatrix(&mut self, _: &(), _: &[f32], _: usize) -> Result<()> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _: &crate::CpuStorage,
        _: &[f32],
        _: usize,
    ) -> Result<()> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn quantize_onto(&mut self, _: &crate::CpuStorage) -> Result<()> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(&self, _: &crate::Shape, _: &(), _: &crate::Layout) -> Result<((), crate::Shape)> {
        Err(Error::msg("not compiled with wgpu support"))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::msg("not compiled with wgpu support"))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _: &(),
    _: &[T],
) -> Result<super::QStorage> {
    let _ = std::any::type_name::<T>();
    Err(Error::msg("not compiled with wgpu support"))
}
