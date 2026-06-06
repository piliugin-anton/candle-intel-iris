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

use candle_core::quantized::k_quants::{self, BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ8_0};
use candle_core::quantized::GgmlType;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::wgpu_device::{
    dispatch_dequant_f32, dispatch_qmatmul_q4_0, dispatch_qmatmul_q4_k, dispatch_qmatmul_q5_0,
    dispatch_qmatmul_q8_0, upload_quant_weights,
};
use candle_core::{backend::BackendStorage, Device, DType, IndexOp, Module, Result, Storage, Tensor};

const EPS: f32 = 1e-4;
const BF16_EPS: f32 = 0.03;
const F16_EPS: f32 = 5e-3;

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

fn assert_qmatmul_parity_tol(cpu: &Tensor, gpu: &Tensor, tol: f32) -> Result<()> {
    let d = max_abs_diff(cpu, gpu)?;
    assert!(d < tol, "qmatmul max abs diff {d} >= {tol}");
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
    let d = max_abs_diff(&out_cpu.to_dtype(DType::F32)?, &out_gpu.to_dtype(DType::F32)?)?;
    assert!(d < 2e-4, "f16 matmul max abs diff {d} >= 2e-4");

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
    let rhs_buf = upload_quant_weights(&wgpu_dev, bytes)?;
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
fn parity_qmatmul_q5_0_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let wgpu_dev = match &gpu {
        Device::Wgpu(d) => d.clone(),
        _ => unreachable!(),
    };

    let m = 2usize;
    let n = 4usize;
    let k = 64usize;
    let lhs_s: Vec<f32> = (0..m * k).map(|v| (v as f32) * 0.01 - 0.5).collect();
    let rhs_s: Vec<f32> = (0..n * k).map(|v| (v as f32) * 0.02 - 1.0).collect();
    let mut rhs_blocks = vec![BlockQ5_0::zeros(); n * k / 32];
    BlockQ5_0::from_float(&rhs_s, &mut rhs_blocks);
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
            rhs_blocks.len() * std::mem::size_of::<BlockQ5_0>(),
        )
    };
    let rhs_buf = upload_quant_weights(&wgpu_dev, bytes)?;
    let out_gpu = dispatch_qmatmul_q5_0(
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
fn parity_unary_trig_erf_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let xs_cpu = Tensor::from_vec(
        vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0],
        (2, 3),
        &cpu,
    )?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;

    assert_parity(&xs_cpu.sin()?, &xs_gpu.sin()?)?;
    assert_parity(&xs_cpu.cos()?, &xs_gpu.cos()?)?;
    assert_parity(&xs_cpu.tanh()?, &xs_gpu.tanh()?)?;
    assert_parity(&xs_cpu.sqr()?, &xs_gpu.sqr()?)?;
    assert_parity(&xs_cpu.erf()?, &xs_gpu.erf()?)?;
    assert_parity(&xs_cpu.gelu_erf()?, &xs_gpu.gelu_erf()?)?;
    assert_parity(&xs_cpu.sign()?, &xs_gpu.sign()?)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_argmin_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let xs_cpu = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], (2, 3), &cpu)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    assert_eq!(
        xs_cpu.argmin(1)?.to_vec1::<u32>()?,
        xs_gpu.argmin(1)?.to_vec1::<u32>()?,
    );

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

fn assert_parity_tol(cpu: &Tensor, gpu: &Tensor, tol: f32) -> Result<()> {
    let d = max_abs_diff(cpu, gpu)?;
    assert!(d < tol, "max abs diff {d} >= {tol}");
    Ok(())
}

/// CPU f32 reference for fused SDPA parity (q/k/v may be f16/bf16; compared in f32).
fn sdpa_reference_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let qf = q.to_dtype(DType::F32)?;
    let kf = k.to_dtype(DType::F32)?;
    let vf = v.to_dtype(DType::F32)?;
    let mut att = (qf * scale as f64)?.matmul(&kf.t()?)?;
    if let Some(mask) = causal_mask {
        att = att.broadcast_add(mask)?;
    }
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&vf)
}

fn causal_mask_f32(q_seq: usize, k_seq: usize, device: &Device) -> Result<Tensor> {
    let offset = k_seq - q_seq;
    let mask: Vec<f32> = (0..q_seq)
        .flat_map(|i| {
            (0..k_seq).map(move |j| {
                if j as isize <= i as isize + offset as isize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, q_seq, k_seq), device)
}

fn assert_sdpa_low_precision_parity(
    cpu: &Device,
    gpu: &Device,
    bs: usize,
    heads: usize,
    q_seq: usize,
    k_seq: usize,
    dk: usize,
    dtype: DType,
    do_causal: bool,
    tol: f32,
) -> Result<()> {
    let scale = (dk as f32).sqrt().recip();
    let q = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, q_seq, dk), cpu)?.to_dtype(dtype)?;
    let k = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), cpu)?.to_dtype(dtype)?;
    let v = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), cpu)?.to_dtype(dtype)?;
    let mask = if do_causal {
        Some(causal_mask_f32(q_seq, k_seq, cpu)?)
    } else {
        None
    };
    let truth = sdpa_reference_f32(&q, &k, &v, scale, mask.as_ref())?;
    let fused = candle_nn::ops::sdpa(
        &q.to_device(gpu)?,
        &k.to_device(gpu)?,
        &v.to_device(gpu)?,
        None,
        do_causal,
        scale,
        1.,
    )?;
    assert_parity_tol(&truth, &fused.to_dtype(DType::F32)?, tol)
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
fn parity_rms_norm_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 8), &cpu)?.to_dtype(DType::F16)?;
    let alpha = Tensor::ones(8, DType::F16, &cpu)?;
    let cpu_out = candle_nn::ops::rms_norm(&xs, &alpha, 1e-5)?;
    let gpu_out = candle_nn::ops::rms_norm(&xs.to_device(&gpu)?, &alpha.to_device(&gpu)?, 1e-5)?;
    assert_parity_tol(&cpu_out.to_dtype(DType::F32)?, &gpu_out.to_dtype(DType::F32)?, F16_EPS)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_layer_norm_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 8), &cpu)?;
    let alpha = Tensor::ones(8, DType::F32, &cpu)?;
    let beta = Tensor::zeros(8, DType::F32, &cpu)?;
    let cpu_out = candle_nn::ops::layer_norm(&xs, &alpha, &beta, 1e-5)?;
    let gpu_out = candle_nn::ops::layer_norm(
        &xs.to_device(&gpu)?,
        &alpha.to_device(&gpu)?,
        &beta.to_device(&gpu)?,
        1e-5,
    )?;
    assert_parity(&cpu_out, &gpu_out)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_arg_sort_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::new(&[3.0f32, 1.0, 2.0, 0.5, 4.0, 1.5], &cpu)?.reshape((2, 3))?;
    let cpu_out = xs.arg_sort_last_dim(true)?;
    let gpu_out = xs.to_device(&gpu)?.arg_sort_last_dim(true)?;
    assert_eq!(
        cpu_out.to_vec1::<u32>()?,
        gpu_out.to_vec1::<u32>()?,
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sigmoid_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-2.0f32, 2.0f32, (2, 8), &cpu)?;
    let cpu_out = candle_nn::ops::sigmoid(&xs)?;
    let gpu_out = candle_nn::ops::sigmoid(&xs.to_device(&gpu)?)?;
    assert_parity(&cpu_out, &gpu_out)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_rms_norm_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 8), &cpu)?.to_dtype(DType::BF16)?;
    let alpha = Tensor::ones(8, DType::BF16, &cpu)?;
    let cpu_out = candle_nn::ops::rms_norm(&xs, &alpha, 1e-5)?;
    let gpu_out = candle_nn::ops::rms_norm(&xs.to_device(&gpu)?, &alpha.to_device(&gpu)?, 1e-5)?;
    assert_parity_tol(&cpu_out.to_dtype(DType::F32)?, &gpu_out.to_dtype(DType::F32)?, BF16_EPS)?;
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
fn parity_rope_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 4, 8), &cpu)?.to_dtype(DType::F16)?;
    let cos = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?.to_dtype(DType::F16)?;
    let sin = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?.to_dtype(DType::F16)?;
    let cpu_out = candle_nn::rotary_emb::rope(&xs, &cos, &sin)?;
    let gpu_out = candle_nn::rotary_emb::rope(
        &xs.to_device(&gpu)?,
        &cos.to_device(&gpu)?,
        &sin.to_device(&gpu)?,
    )?;
    assert_parity_tol(&cpu_out.to_dtype(DType::F32)?, &gpu_out.to_dtype(DType::F32)?, F16_EPS)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_rope_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 4, 8), &cpu)?.to_dtype(DType::BF16)?;
    let cos = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?.to_dtype(DType::BF16)?;
    let sin = Tensor::rand(-1.0f32, 1.0f32, (4, 4), &cpu)?.to_dtype(DType::BF16)?;
    let cpu_out = candle_nn::rotary_emb::rope(&xs, &cos, &sin)?;
    let gpu_out = candle_nn::rotary_emb::rope(
        &xs.to_device(&gpu)?,
        &cos.to_device(&gpu)?,
        &sin.to_device(&gpu)?,
    )?;
    assert_parity_tol(&cpu_out.to_dtype(DType::F32)?, &gpu_out.to_dtype(DType::F32)?, BF16_EPS)?;
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

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_last_dim_fused() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (2, 16), &cpu)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    let out_cpu = candle_nn::ops::softmax_last_dim(&xs_cpu)?;
    let out_gpu = candle_nn::ops::softmax_last_dim(&xs_gpu)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_last_dim_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (2, 16), &cpu)?.to_dtype(DType::F16)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    let out_cpu = candle_nn::ops::softmax_last_dim(&xs_cpu)?;
    let out_gpu = candle_nn::ops::softmax_last_dim(&xs_gpu)?;
    assert_parity_tol(
        &out_cpu.to_dtype(DType::F32)?,
        &out_gpu.to_dtype(DType::F32)?,
        F16_EPS,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_softmax_last_dim_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (2, 16), &cpu)?.to_dtype(DType::BF16)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    let out_cpu = candle_nn::ops::softmax_last_dim(&xs_cpu)?;
    let out_gpu = candle_nn::ops::softmax_last_dim(&xs_gpu)?;
    assert_parity_tol(
        &out_cpu.to_dtype(DType::F32)?,
        &out_gpu.to_dtype(DType::F32)?,
        BF16_EPS,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_qmatmul_q8_0_f32() -> Result<()> {
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
    let mut rhs_blocks = vec![BlockQ8_0::zeros(); n * k / 32];
    BlockQ8_0::from_float(&rhs_s, &mut rhs_blocks);
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
            rhs_blocks.len() * std::mem::size_of::<BlockQ8_0>(),
        )
    };
    let rhs_buf = upload_quant_weights(&wgpu_dev, bytes)?;
    let out_gpu = dispatch_qmatmul_q8_0(&wgpu_lhs, &rhs_buf, (1, m, n, k), lhs_layout)?;
    let out_gpu_t = Tensor::from_vec(
        match out_gpu.to_cpu_storage()? {
            candle_core::CpuStorage::F32(v) => v,
            _ => unreachable!(),
        },
        (m, n),
        &cpu,
    )?;
    assert_qmatmul_parity_tol(&out_cpu, &out_gpu_t, 0.001)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_qmatmul_q4_k_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let wgpu_dev = match &gpu {
        Device::Wgpu(d) => d.clone(),
        _ => unreachable!(),
    };

    let m = 2usize;
    let n = 2usize;
    let k = 256usize;
    let lhs_s: Vec<f32> = (0..m * k).map(|v| (v as f32) * 0.001 - 0.5).collect();
    let rhs_s: Vec<f32> = (0..n * k).map(|v| (v as f32) * 0.002 - 1.0).collect();
    let mut rhs_blocks = vec![BlockQ4K::zeros(); n * k / 256];
    BlockQ4K::from_float(&rhs_s, &mut rhs_blocks);
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
            rhs_blocks.len() * std::mem::size_of::<BlockQ4K>(),
        )
    };
    let rhs_buf = upload_quant_weights(&wgpu_dev, bytes)?;
    let out_gpu = dispatch_qmatmul_q4_k(&wgpu_lhs, &rhs_buf, (1, m, n, k), lhs_layout)?;
    let out_gpu_t = Tensor::from_vec(
        match out_gpu.to_cpu_storage()? {
            candle_core::CpuStorage::F32(v) => v,
            _ => unreachable!(),
        },
        (m, n),
        &cpu,
    )?;
    assert_qmatmul_parity_tol(&out_cpu, &out_gpu_t, 0.0025)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_dequant_q4_0_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let wgpu_dev = match &gpu {
        Device::Wgpu(d) => d.clone(),
        _ => unreachable!(),
    };

    let k = 64usize;
    let src = Tensor::rand(-2.0f32, 2.0f32, (k,), &cpu)?;
    let qtensor = QTensor::quantize_onto(&src, GgmlDType::Q4_0, &cpu)?;
    let dequant_cpu = qtensor.dequantize(&cpu)?;

    let bytes = qtensor.data()?;
    let quant_buf = upload_quant_weights(&wgpu_dev, &bytes)?;
    let out_gpu = dispatch_dequant_f32(&wgpu_dev, GgmlDType::Q4_0, &quant_buf, k)?;
    let out_gpu_t = Tensor::from_vec(
        match out_gpu.to_cpu_storage()? {
            candle_core::CpuStorage::F32(v) => v,
            _ => unreachable!(),
        },
        (k,),
        &cpu,
    )?;
    assert_parity(&dequant_cpu, &out_gpu_t)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_dequant_q4_k_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let wgpu_dev = match &gpu {
        Device::Wgpu(d) => d.clone(),
        _ => unreachable!(),
    };

    let k = 256usize;
    let src = Tensor::rand(-2.0f32, 2.0f32, (k,), &cpu)?;
    let qtensor = QTensor::quantize_onto(&src, GgmlDType::Q4K, &cpu)?;
    let dequant_cpu = qtensor.dequantize(&cpu)?;

    let bytes = qtensor.data()?;
    let quant_buf = upload_quant_weights(&wgpu_dev, &bytes)?;
    let out_gpu = dispatch_dequant_f32(&wgpu_dev, GgmlDType::Q4K, &quant_buf, k)?;
    let out_gpu_t = Tensor::from_vec(
        match out_gpu.to_cpu_storage()? {
            candle_core::CpuStorage::F32(v) => v,
            _ => unreachable!(),
        },
        (k,),
        &cpu,
    )?;
    assert_parity(&dequant_cpu, &out_gpu_t)?;

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_quant_dequant_q4_0_roundtrip() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;

    let k = 64usize;
    let src = Tensor::rand(-2.0f32, 2.0f32, (k,), &cpu)?;
    let src_gpu = src.to_device(&gpu)?;
    let qtensor = QTensor::quantize(&src_gpu, GgmlDType::Q4_0)?;
    let roundtrip = qtensor.dequantize(&gpu)?;
    let d = max_abs_diff(&src, &roundtrip.to_device(&cpu)?)?;
    assert!(d < 1.0, "q4_0 gpu quant/dequant max abs diff {d} >= 1.0");

    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_cast_bf16_roundtrip() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs = Tensor::from_vec(vec![1.0f32, -2.5, 0.0, 3.25, -0.125], (5,), &cpu)?;
    let roundtrip = xs.to_device(&gpu)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    assert_parity(&xs, &roundtrip.to_device(&cpu)?)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_matmul_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let a_cpu = Tensor::rand(-1.0f32, 1.0f32, (4, 32), &cpu)?.to_dtype(DType::BF16)?;
    let b_cpu = Tensor::rand(-1.0f32, 1.0f32, (32, 32), &cpu)?.to_dtype(DType::BF16)?;
    let out_cpu = a_cpu
        .to_dtype(DType::F32)?
        .matmul(&b_cpu.to_dtype(DType::F32)?)?;
    let out_gpu = a_cpu.to_device(&gpu)?.matmul(&b_cpu.to_device(&gpu)?)?;
    let d = max_abs_diff(&out_cpu, &out_gpu.to_dtype(DType::F32)?)?;
    assert!(d < BF16_EPS, "bf16 matmul max abs diff {d} >= {BF16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_gelu_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-2.0f32, 2.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let d = max_abs_diff(
        &xs.gelu()?.to_dtype(DType::F32)?,
        &xs.to_device(&gpu)?.gelu()?.to_dtype(DType::F32)?,
    )?;
    assert!(d < BF16_EPS, "bf16 gelu max abs diff {d} >= {BF16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_add_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let a = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let b = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let d = max_abs_diff(
        &a.add(&b)?.to_dtype(DType::F32)?,
        &a.to_device(&gpu)?.add(&b.to_device(&gpu)?)?.to_dtype(DType::F32)?,
    )?;
    assert!(d < BF16_EPS, "bf16 add max abs diff {d} >= {BF16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_silu_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let xs = Tensor::rand(-2.0f32, 2.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let d = max_abs_diff(
        &xs.silu()?.to_dtype(DType::F32)?,
        &xs.to_device(&gpu)?.silu()?.to_dtype(DType::F32)?,
    )?;
    assert!(d < BF16_EPS, "bf16 silu max abs diff {d} >= {BF16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_mul_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let a = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let b = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::BF16)?;
    let d = max_abs_diff(
        &a.mul(&b)?.to_dtype(DType::F32)?,
        &a.to_device(&gpu)?.mul(&b.to_device(&gpu)?)?.to_dtype(DType::F32)?,
    )?;
    assert!(d < BF16_EPS, "bf16 mul max abs diff {d} >= {BF16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_add_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let a = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::F16)?;
    let b = Tensor::rand(-1.0f32, 1.0f32, (2, 8), &cpu)?.to_dtype(DType::F16)?;
    let d = max_abs_diff(
        &a.add(&b)?.to_dtype(DType::F32)?,
        &a.to_device(&gpu)?.add(&b.to_device(&gpu)?)?.to_dtype(DType::F32)?,
    )?;
    assert!(d < F16_EPS, "f16 add max abs diff {d} >= {F16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_vector_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let bs = 2usize;
    let heads = 4usize;
    let q_seq = 1usize;
    let k_seq = 8usize;
    let dk = 64usize;
    let scale = (dk as f32).sqrt().recip();
    let q = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, q_seq, dk), &cpu)?;
    let k = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let v = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let truth = {
        let att = (q.clone() * scale as f64)?.matmul(&k.t()?)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v)?
    };
    let fused = candle_nn::ops::sdpa(
        &q.to_device(&gpu)?,
        &k.to_device(&gpu)?,
        &v.to_device(&gpu)?,
        None,
        false,
        scale,
        1.,
    )?;
    assert_parity(&truth, &fused.to_device(&cpu)?)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_full_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let bs = 2usize;
    let heads = 4usize;
    let q_seq = 16usize;
    let k_seq = 16usize;
    let dk = 64usize;
    let scale = (dk as f32).sqrt().recip();
    let q = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, q_seq, dk), &cpu)?;
    let k = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let v = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let truth = {
        let att = (q.clone() * scale as f64)?.matmul(&k.t()?)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v)?
    };
    let fused = candle_nn::ops::sdpa(
        &q.to_device(&gpu)?,
        &k.to_device(&gpu)?,
        &v.to_device(&gpu)?,
        None,
        false,
        scale,
        1.,
    )?;
    let diff = max_abs_diff(&truth, &fused.to_device(&cpu)?)?;
    assert!(diff < 0.05, "sdpa full max abs diff {diff}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_causal_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let bs = 2usize;
    let heads = 4usize;
    let q_seq = 8usize;
    let k_seq = 8usize;
    let dk = 64usize;
    let scale = (dk as f32).sqrt().recip();
    let q = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, q_seq, dk), &cpu)?;
    let k = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let v = Tensor::rand(-1.0f32, 1.0f32, (bs, heads, k_seq, dk), &cpu)?;
    let offset = k_seq - q_seq;
    let mask: Vec<f32> = (0..q_seq)
        .flat_map(|i| {
            (0..k_seq).map(move |j| {
                if j as isize <= i as isize + offset as isize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let truth = {
        let att = (q.clone() * scale as f64)?.matmul(&k.t()?)?;
        let mask = Tensor::from_vec(mask, (1, 1, q_seq, k_seq), &cpu)?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        att.matmul(&v)?
    };
    let fused = candle_nn::ops::sdpa(
        &q.to_device(&gpu)?,
        &k.to_device(&gpu)?,
        &v.to_device(&gpu)?,
        None,
        true,
        scale,
        1.,
    )?;
    let diff = max_abs_diff(&truth, &fused.to_device(&cpu)?)?;
    assert!(diff < 0.05, "sdpa causal max abs diff {diff}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_vector_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 1, 8, 64, DType::F16, false, 0.05)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_vector_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 1, 8, 64, DType::BF16, false, 0.08)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_full_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 16, 16, 64, DType::F16, false, 0.08)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_full_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 16, 16, 64, DType::BF16, false, 0.08)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_causal_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 8, 8, 64, DType::F16, true, 0.08)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_sdpa_causal_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    assert_sdpa_low_precision_parity(&cpu, &gpu, 2, 4, 8, 8, 64, DType::BF16, true, 0.08)
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv2d_tiny_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    // 1x1 spatial, 1 channel, 2x2 kernel, no padding: single output = dot(input, kernel)
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &cpu)?;
    let w = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, 2), &cpu)?;
    let out_cpu = t.conv2d(&w, 0, 1, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv2d(&w.to_device(&gpu)?, 0, 1, 1, 1)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_matmul_conv_shapes_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    // Shapes matching conv2d im2col+matmul (m=64, k=27, n=4).
    let a_cpu = Tensor::rand(-1.0f32, 1.0f32, (1, 64, 27), &cpu)?;
    let b_cpu = Tensor::rand(-1.0f32, 1.0f32, (1, 27, 4), &cpu)?;
    let out_cpu = a_cpu.matmul(&b_cpu)?;
    let out_gpu = a_cpu.to_device(&gpu)?.matmul(&b_cpu.to_device(&gpu)?)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 3, 8, 8), &cpu)?;
    let w = Tensor::rand(-1.0f32, 1.0f32, (4, 3, 3, 3), &cpu)?;
    let out_cpu = t.conv2d(&w, 1, 1, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv2d(&w.to_device(&gpu)?, 1, 1, 1, 1)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv1d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 16), &cpu)?;
    let w = Tensor::rand(-1.0f32, 1.0f32, (3, 4, 3), &cpu)?;
    let out_cpu = t.conv1d(&w, 0, 1, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv1d(&w.to_device(&gpu)?, 0, 1, 1, 1)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_avg_pool2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t_cpu = Tensor::from_vec(data.clone(), (1, 1, 4, 4), &cpu)?;
    let t_gpu = Tensor::from_vec(data, (1, 1, 4, 4), &gpu)?;
    let out_cpu = t_cpu.avg_pool2d(2)?;
    let out_gpu = t_gpu.avg_pool2d(2)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_max_pool2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let data: Vec<f32> = vec![
        1., 2., 1., 3., 0., 0., 1., 1., 1., 1., 1., 1., 5., 1., 1., 1.,
    ];
    let t_cpu = Tensor::from_vec(data.clone(), (1, 1, 4, 4), &cpu)?;
    let t_gpu = Tensor::from_vec(data, (1, 1, 4, 4), &gpu)?;
    let out_cpu = t_cpu.max_pool2d(2)?;
    let out_gpu = t_gpu.max_pool2d(2)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_upsample_nearest2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 3, 4), &cpu)?;
    let out_cpu = t.upsample_nearest2d(6, 8)?;
    let out_gpu = t.to_device(&gpu)?.upsample_nearest2d(6, 8)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_upsample_bilinear2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 1, 4, 4), &cpu)?;
    let out_cpu = t.upsample_bilinear2d(8, 8, false)?;
    let out_gpu = t.to_device(&gpu)?.upsample_bilinear2d(8, 8, false)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv_transpose2d_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 4, 4), &cpu)?;
    let w = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 3, 3), &cpu)?;
    let out_cpu = t.conv_transpose2d(&w, 0, 0, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv_transpose2d(&w.to_device(&gpu)?, 0, 0, 1, 1)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_index_select_f32_u32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let ids = Tensor::new(&[0u32, 2u32, 1u32], &cpu)?;
    let t = Tensor::arange(0f32, 12f32, &cpu)?.reshape((4, 3))?;
    let out_cpu = t.index_select(&ids, 0)?;
    let out_gpu = t.to_device(&gpu)?.index_select(&ids.to_device(&gpu)?, 0)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_gather_f32_u32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::arange(0f32, 12f32, &cpu)?.reshape((4, 3))?;
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32], &cpu)?.reshape((2, 2))?;
    let out_cpu = t.gather(&ids, 0)?;
    let out_gpu = t.to_device(&gpu)?.gather(&ids.to_device(&gpu)?, 0)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_index_add_f32_u32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let ids = Tensor::new(&[0u32, 1u32, 1u32], &cpu)?;
    let t = Tensor::arange(0f32, 12f32, &cpu)?.reshape((4, 3))?;
    let init = Tensor::ones((4, 2), DType::F32, &cpu)?;
    let out_cpu = init.index_add(&ids, &t, 1)?;
    let out_gpu = init
        .to_device(&gpu)?
        .index_add(&ids.to_device(&gpu)?, &t.to_device(&gpu)?, 1)?;
    assert_parity(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_const_set_u32_strided() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let tensor_cpu = Tensor::zeros((3, 4), DType::U32, &cpu)?;
    tensor_cpu.const_set(42u32.into())?;
    let tensor_gpu = Tensor::zeros((3, 4), DType::U32, &gpu)?;
    tensor_gpu.const_set(42u32.into())?;
    assert_eq!(
        tensor_cpu.to_vec2::<u32>()?,
        tensor_gpu.to_vec2::<u32>()?
    );
    tensor_cpu.i((.., 2))?.const_set(1337u32.into())?;
    tensor_gpu.i((.., 2))?.const_set(1337u32.into())?;
    assert_eq!(
        tensor_cpu.to_vec2::<u32>()?,
        tensor_gpu.to_vec2::<u32>()?
    );
    tensor_cpu.i((2, ..))?.const_set(1u32.into())?;
    tensor_gpu.i((2, ..))?.const_set(1u32.into())?;
    assert_eq!(
        tensor_cpu.to_vec2::<u32>()?,
        tensor_gpu.to_vec2::<u32>()?
    );
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_cmp_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &cpu)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], &cpu)?;
    let g1 = t1.to_device(&gpu)?;
    let g2 = t2.to_device(&gpu)?;
    assert_eq!(t1.eq(&t2)?.to_vec2::<u8>()?, g1.eq(&g2)?.to_vec2::<u8>()?);
    assert_eq!(t1.ne(&t2)?.to_vec2::<u8>()?, g1.ne(&g2)?.to_vec2::<u8>()?);
    assert_eq!(t1.le(&t2)?.to_vec2::<u8>()?, g1.le(&g2)?.to_vec2::<u8>()?);
    assert_eq!(t1.lt(&t2)?.to_vec2::<u8>()?, g1.lt(&g2)?.to_vec2::<u8>()?);
    assert_eq!(t1.gt(&t2)?.to_vec2::<u8>()?, g1.gt(&g2)?.to_vec2::<u8>()?);
    assert_eq!(t1.ge(&t2)?.to_vec2::<u8>()?, g1.ge(&g2)?.to_vec2::<u8>()?);
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_powf_elu_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::new(&[-1.5f32, 0.0, 1.5, 2.0], &cpu)?;
    let cpu_pow = t.powf(2.0)?;
    let gpu_pow = t.to_device(&gpu)?.powf(2.0)?;
    assert_parity(&cpu_pow, &gpu_pow)?;
    let cpu_elu = t.elu(1.1)?;
    let gpu_elu = t.to_device(&gpu)?.elu(1.1)?;
    assert_parity(&cpu_elu, &gpu_elu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_rand_f32() -> Result<()> {
    let gpu = wgpu_device()?;
    gpu.set_seed(42)?;
    let t1 = Tensor::rand(0f32, 1f32, (5, 3), &gpu)?;
    let t2 = Tensor::rand(0f32, 1f32, (5, 3), &gpu)?;
    assert_ne!(t1.to_vec2::<f32>()?, t2.to_vec2::<f32>()?);
    assert_eq!(t1.dims(), [5, 3]);

    let n1 = Tensor::randn(0f32, 1f32, (5, 3), &gpu)?;
    let n2 = Tensor::randn(0f32, 1f32, (5, 3), &gpu)?;
    assert_ne!(n1.to_vec2::<f32>()?, n2.to_vec2::<f32>()?);

    const N: usize = 2;
    let v = (0..100)
        .map(|_| Tensor::randn(0f32, 1f32, N, &gpu).and_then(|t| t.to_vec1::<f32>()))
        .collect::<Result<Vec<_>>>()?;
    assert!(
        (0..N).all(|i| v.windows(2).any(|pair| pair[0][i] != pair[1][i])),
        "deterministic randn values detected"
    );
    Ok(())
}

fn assert_parity_f16(cpu: &Tensor, gpu: &Tensor) -> Result<()> {
    let cpu_f32 = cpu.to_dtype(DType::F32)?;
    let gpu_f32 = gpu.to_dtype(DType::F32)?;
    let d = max_abs_diff(&cpu_f32, &gpu_f32)?;
    assert!(d < F16_EPS, "f16 parity max abs diff {d} >= {F16_EPS}");
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv2d_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 3, 8, 8), &cpu)?.to_dtype(DType::F16)?;
    let w = Tensor::rand(-1.0f32, 1.0f32, (4, 3, 3, 3), &cpu)?.to_dtype(DType::F16)?;
    let out_cpu = t.conv2d(&w, 1, 1, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv2d(&w.to_device(&gpu)?, 1, 1, 1, 1)?;
    assert_parity_f16(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_reduce_sum_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (2, 4, 6), &cpu)?.to_dtype(DType::F16)?;
    let out_cpu = t.sum(1)?;
    let out_gpu = t.to_device(&gpu)?.sum(1)?;
    assert_parity_f16(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_avg_pool2d_f16() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 2, 8, 8), &cpu)?.to_dtype(DType::F16)?;
    let out_cpu = t.avg_pool2d(2)?;
    let out_gpu = t.to_device(&gpu)?.avg_pool2d(2)?;
    assert_parity_f16(&out_cpu, &out_gpu)?;
    Ok(())
}

#[test]
#[ignore = "requires GPU with wgpu backend"]
fn parity_conv2d_bf16() -> Result<()> {
    let gpu = wgpu_device()?;
    if !gpu.supports_bf16() {
        return Ok(());
    }
    let cpu = Device::Cpu;
    let t = Tensor::rand(-1.0f32, 1.0f32, (1, 3, 8, 8), &cpu)?.to_dtype(DType::BF16)?;
    let w = Tensor::rand(-1.0f32, 1.0f32, (4, 3, 3, 3), &cpu)?.to_dtype(DType::BF16)?;
    let out_cpu = t
        .to_dtype(DType::F32)?
        .conv2d(&w.to_dtype(DType::F32)?, 1, 1, 1, 1)?;
    let out_gpu = t
        .to_device(&gpu)?
        .conv2d(&w.to_device(&gpu)?, 1, 1, 1, 1)?;
    let d = max_abs_diff(&out_cpu.to_dtype(DType::F32)?, &out_gpu.to_dtype(DType::F32)?)?;
    assert!(d < BF16_EPS, "bf16 conv2d max abs diff {d} >= {BF16_EPS}");
    Ok(())
}
