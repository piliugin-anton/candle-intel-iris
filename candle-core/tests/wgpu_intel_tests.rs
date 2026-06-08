//! Integration tests for Intel Iris wgpu optimizations.
//!
//! Run with: `cargo test -p candle-core --features wgpu -- --ignored wgpu_intel`

#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Shape, Tensor};

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn mapped_round_trip() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
    let cpu = t.to_vec1::<f32>()?;
    assert_eq!(cpu, vec![1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn to_dtype_f16_on_gen11() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let t = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &device)?;
    let f16 = t.to_dtype(DType::F16)?;
    assert_eq!(f16.dtype(), DType::F16);
    let back = f16.to_dtype(DType::F32)?;
    let vals = back.to_vec1::<f32>()?;
    assert!((vals[0] - 1.0).abs() < 1e-3);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn pinned_mapped_alloc() -> candle_core::Result<()> {
    use candle_core::WgpuDevice;
    let wgpu = WgpuDevice::new_default().map_err(candle_core::Error::msg)?;
    let storage = wgpu.alloc_pinned_mapped(&Shape::from((4,)), DType::F32)?;
    assert!(storage.is_mapped());
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_layout_readback_reads_slice_only() -> candle_core::Result<()> {
    use candle_core::{Device, IndexOp, Tensor};

    let device = Device::new_wgpu()?;
    let t = Tensor::from_vec((0..20).map(|v| v as f32).collect(), (20,), &device)?;
    let slice = t.i((9..12,))?;
    assert_eq!(slice.to_vec1::<f32>()?, vec![9.0, 10.0, 11.0]);
    let scalar = t.i((15,))?;
    assert_eq!(scalar.to_scalar::<f32>()?, 15.0);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_add_f32() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (3,), &device)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_exp_f32() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let a = Tensor::from_vec(vec![0.0f32, 1.0], (2,), &device)?;
    let e = a.exp()?;
    let vals = e.to_vec1::<f32>()?;
    assert!((vals[0] - 1.0).abs() < 1e-5);
    assert!((vals[1] - std::f32::consts::E).abs() < 1e-4);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_matmul_f32() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;
    let b = Tensor::from_vec(
        vec![
            7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        (3, 4),
        &device,
    )?;
    let c = a.matmul(&b)?;
    let expected = a
        .to_device(&Device::Cpu)?
        .matmul(&b.to_device(&Device::Cpu)?)?;
    let diff = c
        .to_device(&Device::Cpu)?
        .sub(&expected)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-3, "matmul diff {diff}");
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_max_f32() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let a = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 2.0], (2, 2), &device)?;
    let m = a.max(1)?;
    assert_eq!(m.to_vec1::<f32>()?, vec![5.0, 3.0]);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_sum_f32() -> candle_core::Result<()> {
    let device = Device::new_wgpu()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let s = a.sum(1)?;
    assert_eq!(s.to_vec1::<f32>()?, vec![3.0, 7.0]);
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_narrow_view_unaligned_offset() -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_wgpu()?;

    let base_cpu = Tensor::from_vec((0..20).map(|i| i as f32).collect::<Vec<_>>(), (1, 20), &cpu)?;
    let a_cpu = base_cpu.narrow(1, 9, 3)?;
    let b_cpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &cpu)?;
    let add_cpu = a_cpu.add(&b_cpu)?;
    let mul_cpu = a_cpu.mul(&b_cpu)?;

    let base_gpu =
        Tensor::from_vec((0..20).map(|i| i as f32).collect::<Vec<_>>(), (1, 20), &gpu)?;
    let a_gpu = base_gpu.narrow(1, 9, 3)?;
    let b_gpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &gpu)?;
    let add_gpu = a_gpu.add(&b_gpu)?;
    let mul_gpu = a_gpu.mul(&b_gpu)?;

    assert_eq!(add_gpu.to_vec2::<f32>()?, add_cpu.to_vec2::<f32>()?);
    assert_eq!(mul_gpu.to_vec2::<f32>()?, mul_cpu.to_vec2::<f32>()?);
    Ok(())
}
