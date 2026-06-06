//! CPU fallback helpers for custom ops without a native wgpu kernel.

use super::error::Result;
use super::storage::WgpuStorage;
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
use crate::{Layout, Shape};

fn storage_from_cpu(device: &WgpuDevice, cpu: crate::CpuStorage) -> Result<WgpuStorage> {
    WgpuStorage::from_cpu(device, &cpu)}

/// Runs `op` on CPU and uploads the result to wgpu storage.
pub fn cpu_fallback_op1<C: CustomOp1 + ?Sized>(
    op: &C,
    storage: &WgpuStorage,
    layout: &Layout,
) -> Result<(WgpuStorage, Shape)> {
    let cpu = storage.to_cpu_storage().map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let (out, shape) = op
        .cpu_fwd(&cpu, layout)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let wgpu = storage_from_cpu(storage.device(), out)?;
    Ok((wgpu, shape))
}

/// Runs `op` on CPU and uploads the result to wgpu storage.
pub fn cpu_fallback_op2<C: CustomOp2 + ?Sized>(
    op: &C,
    s1: &WgpuStorage,
    l1: &Layout,
    s2: &WgpuStorage,
    l2: &Layout,
) -> Result<(WgpuStorage, Shape)> {
    let cpu1 = s1
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu2 = s2
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let (out, shape) = op
        .cpu_fwd(&cpu1, l1, &cpu2, l2)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let wgpu = storage_from_cpu(s1.device(), out)?;
    Ok((wgpu, shape))
}

/// Runs `op` on CPU and uploads the result to wgpu storage.
pub fn cpu_fallback_op3<C: CustomOp3 + ?Sized>(
    op: &C,
    s1: &WgpuStorage,
    l1: &Layout,
    s2: &WgpuStorage,
    l2: &Layout,
    s3: &WgpuStorage,
    l3: &Layout,
) -> Result<(WgpuStorage, Shape)> {
    let cpu1 = s1
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu2 = s2
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu3 = s3
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let (out, shape) = op
        .cpu_fwd(&cpu1, l1, &cpu2, l2, &cpu3, l3)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let wgpu = storage_from_cpu(s1.device(), out)?;
    Ok((wgpu, shape))
}

/// Runs an in-place `op` on CPU and replaces wgpu storage with the result.
pub fn cpu_fallback_inplace_op1<C: InplaceOp1 + ?Sized>(
    op: &C,
    storage: &mut WgpuStorage,
    layout: &Layout,
) -> Result<()> {
    let mut cpu = storage
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    op.cpu_fwd(&mut cpu, layout)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    *storage = storage_from_cpu(storage.device(), cpu)?;
    Ok(())
}

/// Runs an in-place `op` on CPU and replaces wgpu storage with the result.
pub fn cpu_fallback_inplace_op2<C: InplaceOp2 + ?Sized>(
    op: &C,
    s1: &mut WgpuStorage,
    l1: &Layout,
    s2: &WgpuStorage,
    l2: &Layout,
) -> Result<()> {
    let mut cpu1 = s1
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu2 = s2
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    op.cpu_fwd(&mut cpu1, l1, &cpu2, l2)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    *s1 = storage_from_cpu(s1.device(), cpu1)?;
    Ok(())
}

/// Runs an in-place `op` on CPU and replaces wgpu storage with the result.
pub fn cpu_fallback_inplace_op3<C: InplaceOp3 + ?Sized>(
    op: &C,
    s1: &mut WgpuStorage,
    l1: &Layout,
    s2: &WgpuStorage,
    l2: &Layout,
    s3: &WgpuStorage,
    l3: &Layout,
) -> Result<()> {
    let mut cpu1 = s1
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu2 = s2
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    let cpu3 = s3
        .to_cpu_storage()
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    op.cpu_fwd(&mut cpu1, l1, &cpu2, l2, &cpu3, l3)
        .map_err(|e| super::error::WgpuError::Message(e.to_string()))?;
    *s1 = storage_from_cpu(s1.device(), cpu1)?;
    Ok(())
}
