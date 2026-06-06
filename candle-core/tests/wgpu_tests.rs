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

use candle_core::{Device, Result, Tensor};

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
fn parity_softmax_f32_3d() -> Result<()> {
    let gpu = wgpu_device()?;
    let cpu = Device::Cpu;
    let xs_cpu = Tensor::rand(-3.0f32, 3.0f32, (1, 8, 16), &cpu)?;
    let xs_gpu = xs_cpu.to_device(&gpu)?;
    assert_parity(&softmax(&xs_cpu, 2)?, &softmax(&xs_gpu, 2)?)?;
    Ok(())
}
