//! Integration tests for Intel Iris wgpu optimizations.
//!
//! Run with: `cargo test -p candle-core --features wgpu --test wgpu_intel_tests -- --ignored`

#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};

const F32_EPS: f32 = 1e-3;
const F16_EPS: f32 = 5e-2;
const F16_LARGE_K_EPS: f32 = 0.5;
const BF16_EPS: f32 = 0.15;
const BF16_LARGE_K_EPS: f32 = 1.5;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let cpu = Device::Cpu;
    let a = a.to_dtype(DType::F32)?.to_device(&cpu)?;
    let b = b.to_dtype(DType::F32)?.to_device(&cpu)?;
    a.sub(&b)?.abs()?.max_all()?.to_scalar()
}

fn assert_matmul_parity(
    gpu: &Device,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    tol: f32,
) -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let a_cpu = Tensor::randn(0f32, 1.0, (m, k), &cpu)?.to_dtype(dtype)?;
    let b_cpu = Tensor::randn(0f32, 1.0, (k, n), &cpu)?.to_dtype(dtype)?;
    // Half dtypes on long inner dimensions: compare against f32 CPU reference
    // (same policy as `parity_matmul_bf16` in wgpu_tests.rs).
    let c_cpu = match dtype {
        DType::BF16 => a_cpu
            .to_dtype(DType::F32)?
            .matmul(&b_cpu.to_dtype(DType::F32)?)?,
        DType::F16 if k >= 256 => a_cpu
            .to_dtype(DType::F32)?
            .matmul(&b_cpu.to_dtype(DType::F32)?)?,
        _ => a_cpu.matmul(&b_cpu)?,
    };

    let a_gpu = a_cpu.to_device(gpu)?;
    let b_gpu = b_cpu.to_device(gpu)?;
    let c_gpu = a_gpu.matmul(&b_gpu)?;

    let diff = max_abs_diff(&c_cpu, &c_gpu)?;
    assert!(
        diff < tol,
        "matmul ({m},{k})x({k},{n}) {dtype:?} max diff {diff} >= {tol}"
    );
    Ok(())
}

fn assert_mlp_block_parity(gpu: &Device, dtype: DType, tol: f32) -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    // Token projection: (batch=4, hidden=512) @ (512, 2048) + bias broadcast
    let x_cpu = Tensor::randn(0f32, 1.0, (4, 512), &cpu)?.to_dtype(dtype)?;
    let w_cpu = Tensor::randn(0f32, 0.02, (512, 2048), &cpu)?.to_dtype(dtype)?;
    let b_cpu = Tensor::randn(0f32, 0.01, (2048,), &cpu)?.to_dtype(dtype)?;

    let w2_cpu = Tensor::randn(0f32, 0.02, (2048, 512), &cpu)?.to_dtype(dtype)?;
    let h_cpu = match dtype {
        DType::BF16 => x_cpu
            .to_dtype(DType::F32)?
            .matmul(&w_cpu.to_dtype(DType::F32)?)?
            .broadcast_add(&b_cpu.to_dtype(DType::F32)?.reshape((1, 2048))?)?,
        _ => x_cpu
            .matmul(&w_cpu)?
            .broadcast_add(&b_cpu.reshape((1, 2048))?)?,
    };
    let y_cpu = match dtype {
        DType::F16 | DType::BF16 => h_cpu
            .relu()?
            .to_dtype(DType::F32)?
            .matmul(&w2_cpu.to_dtype(DType::F32)?)?,
        _ => h_cpu.relu()?.matmul(&w2_cpu)?,
    };

    let x_gpu = x_cpu.to_device(gpu)?;
    let w_gpu = w_cpu.to_device(gpu)?;
    let b_gpu = b_cpu.to_device(gpu)?;
    let w2_gpu = w2_cpu.to_device(gpu)?;

    let h_gpu = x_gpu
        .matmul(&w_gpu)?
        .broadcast_add(&b_gpu.reshape((1, 2048))?)?;
    let y_gpu = h_gpu.relu()?.matmul(&w2_gpu)?;

    let mlp_tol = match dtype {
        DType::F16 => F16_LARGE_K_EPS,
        DType::BF16 => BF16_EPS * 2.0,
        _ => tol,
    };
    let diff = max_abs_diff(&y_cpu, &y_gpu)?;
    assert!(diff < mlp_tol, "mlp block {dtype:?} max diff {diff} >= {mlp_tol}");
    Ok(())
}

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
    let storage = wgpu.alloc_pinned_mapped(&candle_core::Shape::from((4,)), DType::F32)?;
    assert!(storage.is_mapped());
    Ok(())
}

#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_layout_readback_reads_slice_only() -> candle_core::Result<()> {
    use candle_core::IndexOp;

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
    let diff = max_abs_diff(&c.to_device(&Device::Cpu)?, &expected)?;
    assert!(diff < F32_EPS, "matmul diff {diff}");
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

/// Transformer-like GEMV matmul shapes (batch=1, large K/N) on real random tensors.
#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_matmul_gemv_shapes_real_tensors() -> candle_core::Result<()> {
    let gpu = Device::new_wgpu()?;
    for &(m, k, n) in &[(1, 4096, 4096), (1, 2048, 8192), (8, 512, 512)] {
        assert_matmul_parity(&gpu, DType::F32, m, k, n, F32_EPS)?;
        let f16_tol = if k >= 256 { F16_LARGE_K_EPS } else { F16_EPS };
        assert_matmul_parity(&gpu, DType::F16, m, k, n, f16_tol)?;
        let bf16_tol = if k >= 256 { BF16_LARGE_K_EPS } else { BF16_EPS };
        assert_matmul_parity(&gpu, DType::BF16, m, k, n, bf16_tol)?;
    }
    Ok(())
}

/// Square tiled matmul on realistic hidden sizes.
#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_matmul_tiled_shapes_real_tensors() -> candle_core::Result<()> {
    let gpu = Device::new_wgpu()?;
    for &(m, k, n) in &[(64, 64, 64), (128, 256, 128), (32, 512, 512)] {
        assert_matmul_parity(&gpu, DType::F32, m, k, n, F32_EPS)?;
        let f16_tol = if k >= 256 { F16_LARGE_K_EPS } else { F16_EPS };
        assert_matmul_parity(&gpu, DType::F16, m, k, n, f16_tol)?;
        let bf16_tol = if k >= 256 { BF16_LARGE_K_EPS } else { BF16_EPS };
        assert_matmul_parity(&gpu, DType::BF16, m, k, n, bf16_tol)?;
    }
    Ok(())
}

/// Mini MLP block: matmul + bias broadcast + relu + matmul (typical FFN pattern).
#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_mlp_block_real_tensors() -> candle_core::Result<()> {
    let gpu = Device::new_wgpu()?;
    assert_mlp_block_parity(&gpu, DType::F32, F32_EPS)?;
    assert_mlp_block_parity(&gpu, DType::F16, F16_EPS)?;
    assert_mlp_block_parity(&gpu, DType::BF16, BF16_EPS)?;
    Ok(())
}

/// Single-token attention projection: (1, hidden) @ (hidden, 3*hidden).
#[test]
#[ignore = "requires Intel GPU with wgpu backend"]
fn wgpu_qkv_projection_real_tensors() -> candle_core::Result<()> {
    let gpu = Device::new_wgpu()?;
    let hidden = 4096;
    let cpu = Device::Cpu;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let x = Tensor::randn(0f32, 1.0, (1, hidden), &cpu)?.to_dtype(dtype)?;
        let w = Tensor::randn(0f32, 0.02, (hidden, hidden * 3), &cpu)?.to_dtype(dtype)?;
        let y_cpu = match dtype {
            DType::F16 | DType::BF16 => x
                .to_dtype(DType::F32)?
                .matmul(&w.to_dtype(DType::F32)?)?,
            _ => x.matmul(&w)?,
        };
        let y_gpu = x.to_device(&gpu)?.matmul(&w.to_device(&gpu)?)?;
        let tol = match dtype {
            DType::F16 => F16_LARGE_K_EPS,
            DType::BF16 => BF16_EPS,
            _ => F32_EPS,
        };
        let diff = max_abs_diff(&y_cpu, &y_gpu)?;
        assert!(diff < tol, "qkv {dtype:?} diff {diff}");
    }
    Ok(())
}
