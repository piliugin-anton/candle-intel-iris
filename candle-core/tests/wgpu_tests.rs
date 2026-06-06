//! CPU vs WGPU parity integration tests.
//!
//! Run (requires GPU):
//! ```text
//! cargo test -p candle-core --features wgpu --test wgpu_tests -- --ignored --test-threads=1
//! ```
//!
//! With wgpu validation layers:
//! ```text
//! WGPU_VALIDATION=1 cargo test -p candle-core --features wgpu --test wgpu_tests -- --ignored --test-threads=1
//! ```
//!
//! ## Profiling (Intel GPA)
//!
//! 1. `cargo build -p candle-core --features wgpu --release --example wgpu_bench`
//! 2. Capture in GPA while running:
//!    `cargo run -p candle-core --features wgpu --release --example wgpu_bench`
//! 3. Inspect memory bandwidth and EU occupancy; kernels with low compute/memory ratio are
//!    often memory-bound on integrated GPUs.
//!
//! ## Profiling (Tracy, manual)
//!
//! Build with tracy-client externally if desired, then capture while running parity tests or
//! `wgpu_bench`. Look for long `dispatch_bind_group` / queue submit stalls.

#![cfg(feature = "wgpu")]

use candle_core::quantized::k_quants::{self, BlockQ4_0};
use candle_core::quantized::GgmlType;
use candle_core::wgpu_device::{dispatch_qmatmul_q4_0, upload_q4_0_weights};
use candle_core::{backend::BackendStorage, Device, DType, Module, Result, Storage, Tensor};

const EPS: f32 = 1e-4;

fn wgpu_device() -> Result<Device> {
    Device::new_wgpu()
}

/// L-inf norm of |cpu - gpu_on_cpu|; works for any shape.
fn max_abs_diff(cpu: &Tensor, gpu: &Tensor) -> Result<f32> {
    let gpu_cpu = gpu.to_device(&Device::Cpu)?;
    cpu.sub(&gpu_cpu)?.abs()?.max_all()?.to_scalar()
}

fn assert_parity(cpu: &Tensor, gpu: &Tensor) -> Result<()> {
    let d = max_abs_diff(cpu, gpu)?;
    assert!(d < EPS, "max abs diff {d} >= {EPS}");
    Ok(())
}

/// Q4_0 matmul tolerates small quantization drift between CPU and GPU paths.
fn assert_qmatmul_parity(cpu: &Tensor, gpu: &Tensor) -> Result<()> {
    let d = max_abs_diff(cpu, gpu)?;
    assert!(d < 64.0, "qmatmul max abs diff {d} >= 64.0");
    Ok(())
}

/// Softmax along `dim` (same composition as `candle_nn::ops::softmax`).
fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_add_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    // Fixed small vectors
    let a_data = vec![1.0f32, 2.0, 3.0];
    let b_data = vec![4.0f32, 5.0, 6.0];
    let a_cpu = Tensor::from_vec(a_data.clone(), (3,), &cpu)?;
    let b_cpu = Tensor::from_vec(b_data.clone(), (3,), &cpu)?;
    let out_cpu = a_cpu.add(&b_cpu)?;
    let a_gpu = Tensor::from_vec(a_data, (3,), &gpu)?;
    let b_gpu = Tensor::from_vec(b_data, (3,), &gpu)?;
    let out_gpu = a_gpu.add(&b_gpu)?;
    assert_parity(&out_cpu, &out_gpu)?;

    // Random 2D
    let a_cpu = Tensor::rand(-2.0f32, 2.0f32, (2, 3), &cpu)?;
    let b_cpu = Tensor::rand(-2.0f32, 2.0f32, (2, 3), &cpu)?;
    let out_cpu = a_cpu.add(&b_cpu)?;
    let a_gpu = a_cpu.to_device(&gpu)?;
    let b_gpu = b_cpu.to_device(&gpu)?;
    let out_gpu = a_gpu.add(&b_gpu)?;
    assert_parity(&out_cpu, &out_gpu)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_matmul_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let a_cpu = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &cpu)?;
    let b_cpu = Tensor::from_vec(
        vec![
            7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        (3, 4),
        &cpu,
    )?;
    let out_cpu = a_cpu.matmul(&b_cpu)?;
    let a_gpu = a_cpu.to_device(&gpu)?;
    let b_gpu = b_cpu.to_device(&gpu)?;
    let out_gpu = a_gpu.matmul(&b_gpu)?;
    assert_parity(&out_cpu, &out_gpu)?;

    let a_cpu = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?;
    let b_cpu = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?;
    let out_cpu = a_cpu.matmul(&b_cpu)?;
    let a_gpu = a_cpu.to_device(&gpu)?;
    let b_gpu = b_cpu.to_device(&gpu)?;
    let out_gpu = a_gpu.matmul(&b_gpu)?;
    assert_parity(&out_cpu, &out_gpu)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_matmul_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let a_cpu = Tensor::rand(-1.0f32, 1.0f32, (4, 32), &cpu)?.to_dtype(DType::F16)?;
    let b_cpu = Tensor::rand(-1.0f32, 1.0f32, (32, 32), &cpu)?.to_dtype(DType::F16)?;
    let out_cpu = a_cpu.matmul(&b_cpu)?;
    let a_gpu = a_cpu.to_device(&gpu)?;
    let b_gpu = b_cpu.to_device(&gpu)?;
    let out_gpu = a_gpu.matmul(&b_gpu)?;
    assert_parity(&out_cpu.to_dtype(DType::F32)?, &out_gpu.to_dtype(DType::F32)?)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_qmatmul_qtensor_q4_0() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let m = 3usize;
    let k = 64usize;
    let n = 4usize;
    let lhs_s: Vec<f32> = (0..m * k).map(|v| v as f32).collect();
    let rhs_s: Vec<f32> = (0..k * n).map(|v| v as f32).collect();
    let lhs = Tensor::from_vec(lhs_s, (m, k), &cpu)?;
    let tensor_rhs = Tensor::from_vec(rhs_s, (n, k), &cpu)?.t()?;
    let weights = tensor_rhs.t()?;
    let out_cpu = {
        let qtensor = candle_core::quantized::QTensor::quantize_onto(
            &weights,
            candle_core::quantized::GgmlDType::Q4_0,
            &cpu,
        )?;
        let matmul = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        matmul.forward(&lhs)?
    };

    let lhs_gpu = lhs.to_device(&gpu)?;
    let qtensor_gpu = candle_core::quantized::QTensor::quantize_onto(
        &weights,
        candle_core::quantized::GgmlDType::Q4_0,
        &gpu,
    )?;
    let matmul_gpu = candle_core::quantized::QMatMul::from_qtensor(qtensor_gpu)?;
    let out_gpu = matmul_gpu.forward(&lhs_gpu)?;

    assert_qmatmul_parity(&out_cpu, &out_gpu)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_qmatmul_q4_0_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let wgpu_dev = match &gpu {
        Device::Wgpu(d) => d.clone(),
        _ => unreachable!(),
    };

    let m = 2usize;
    let n = 4usize;
    let k = 32usize;
    let lhs_s: Vec<f32> = (0..m * k).map(|v| (v as f32) * 0.01 - 0.5).collect();
    let rhs_s: Vec<f32> = (0..n * k).map(|v| (v as f32) * 0.02 - 1.0).collect();
    let mut rhs_blocks = vec![BlockQ4_0::zeros(); n * k / 32];
    BlockQ4_0::from_float(&rhs_s, &mut rhs_blocks);
    let mut dst_cpu = vec![0f32; m * n];
    k_quants::matmul((m, k, n), &lhs_s, &rhs_blocks, &mut dst_cpu)?;
    let out_cpu = Tensor::from_vec(dst_cpu, (m, n), &cpu)?;

    let lhs_gpu = Tensor::from_vec(lhs_s, (m, k), &gpu)?;
    let (lhs_guard, lhs_layout) = lhs_gpu.storage_and_layout();
    let wgpu_lhs = match &*lhs_guard {
        Storage::Wgpu(s) => s.clone(),
        _ => unreachable!(),
    };
    let bytes = unsafe {
        std::slice::from_raw_parts(
            rhs_blocks.as_ptr().cast(),
            rhs_blocks.len() * std::mem::size_of::<BlockQ4_0>(),
        )
    };
    let rhs_buf = upload_q4_0_weights(&wgpu_dev, bytes)?;
    let out_gpu = dispatch_qmatmul_q4_0(
        &wgpu_lhs,
        &rhs_buf,
        (1, m, n, k),
        lhs_layout,
    )?;
    let out_gpu_t = Tensor::from_vec(
        match out_gpu.to_cpu_storage()? {
            candle_core::CpuStorage::F32(v) => v,
            _ => unreachable!(),
        },
        (m, n),
        &cpu,
    )?;
    assert_qmatmul_parity(&out_cpu, &out_gpu_t)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_f32_fixed() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::from_vec(
        vec![0.0f32, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0, -2.0],
        (2, 4),
        &cpu,
    )?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    assert_parity(&softmax(&xs_cpu, 1)?, &softmax(&xs_gpu, 1)?)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_f32_random_2d() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (2, 4), &cpu)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    assert_parity(&softmax(&xs_cpu, 1)?, &softmax(&xs_gpu, 1)?)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_elemwise_128() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let a = Tensor::rand(-1.0f32, 1.0f32, (1, 8, 16), &cpu)?;
    let b = Tensor::rand(-1.0f32, 1.0f32, (1, 8, 16), &cpu)?;
    let ag = a.to_device(&gpu)?;
    let bg = b.to_device(&gpu)?;
    assert_parity(&a.add(&b)?, &ag.add(&bg)?)?;
    assert_parity(&a.sub(&b)?, &ag.sub(&bg)?)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_gelu_affine_reduce_min_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let xs_cpu = Tensor::from_vec(
        vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0],
        (2, 3),
        &cpu,
    )?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    let gelu_cpu = xs_cpu.gelu()?;
    let gelu_gpu = xs_gpu.gelu()?;
    assert_parity(&gelu_cpu, &gelu_gpu)?;

    assert_parity(&xs_cpu.affine(2.0, 0.5)?, &xs_gpu.affine(2.0, 0.5)?)?;

    assert_parity(&xs_cpu.min(1)?, &xs_gpu.min(1)?)?;
    assert_parity(&xs_cpu.max(1)?, &xs_gpu.max(1)?)?;
    assert_parity(&xs_cpu.sum(1)?, &xs_gpu.sum(1)?)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_qmatmul_llm_shape() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let batch = 1usize;
    let seq = 8usize;
    let hidden = 128usize;
    let out = 64usize;
    let lhs_s: Vec<f32> = (0..batch * seq * hidden)
        .map(|v| (v as f32) * 0.001 - 0.5)
        .collect();
    let rhs_s: Vec<f32> = (0..out * hidden)
        .map(|v| (v as f32) * 0.002 - 1.0)
        .collect();
    let lhs = Tensor::from_vec(lhs_s, (batch, seq, hidden), &cpu)?;
    let weights = Tensor::from_vec(rhs_s, (out, hidden), &cpu)?;
    let out_cpu = {
        let qtensor = candle_core::quantized::QTensor::quantize_onto(
            &weights,
            candle_core::quantized::GgmlDType::Q4_0,
            &cpu,
        )?;
        let matmul = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        matmul.forward(&lhs)?
    };

    let lhs_gpu = lhs.to_device(&gpu)?;
    let qtensor_gpu = candle_core::quantized::QTensor::quantize_onto(
        &weights,
        candle_core::quantized::GgmlDType::Q4_0,
        &gpu,
    )?;
    let matmul_gpu = candle_core::quantized::QMatMul::from_qtensor(qtensor_gpu)?;
    let out_gpu = matmul_gpu.forward(&lhs_gpu)?;
    assert_qmatmul_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_rms_norm_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 8), &cpu)?;
    let alpha = Tensor::ones(8, DType::F32, &cpu)?;
    let cpu_out = candle_nn::ops::rms_norm(&xs, &alpha, 1e-5)?;
    let gpu_out = candle_nn::ops::rms_norm(&xs.to_device(&gpu)?, &alpha.to_device(&gpu)?, 1e-5)?;
    assert_parity(&cpu_out, &gpu_out)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_rope_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 4, 8), &cpu)?;
    let cos = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?;
    let sin = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?;
    let cpu_out = candle_nn::rotary_emb::rope(&xs, &cos, &sin)?;
    let gpu_out = candle_nn::rotary_emb::rope(
        &xs.to_device(&gpu)?,
        &cos.to_device(&gpu)?,
        &sin.to_device(&gpu)?,
    )?;
    assert_parity(&cpu_out, &gpu_out)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_cat_kv_cache() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let a = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 4, 8), &cpu)?;
    let b = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 1, 8), &cpu)?;
    let cpu_cat = Tensor::cat(&[&a, &b], 2)?;
    let gpu_cat = Tensor::cat(&[&a.to_device(&gpu)?, &b.to_device(&gpu)?], 2)?;
    assert_parity(&cpu_cat, &gpu_cat)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_f32_3d() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (1, 8, 16), &cpu)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    assert_parity(&softmax(&xs_cpu, 2)?, &softmax(&xs_gpu, 2)?)?;
    Ok(())
}
