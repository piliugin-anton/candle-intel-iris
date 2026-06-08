//! CPU vs WGPU timing comparison for transformer-like tensor shapes.
//!
//! ```text
//! cargo run -p candle-core --features wgpu --release --example wgpu_perf_compare
//! ```

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn bench_matmul(
    device: &Device,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    warmup: u32,
    iters: u32,
) -> candle_core::Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (m, k), device)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1.0, (k, n), device)?.to_dtype(dtype)?;
    for _ in 0..warmup {
        let _ = a.matmul(&b)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = a.matmul(&b)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters))
}

fn bench_binary(
    device: &Device,
    dtype: DType,
    elems: usize,
    warmup: u32,
    iters: u32,
) -> candle_core::Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (elems,), device)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1.0, (elems,), device)?.to_dtype(dtype)?;
    for _ in 0..warmup {
        let _ = a.add(&b)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = a.add(&b)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters))
}

fn main() -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_wgpu()?;
    println!("WGPU device: {gpu:?}\n");

    println!("=== matmul (ms/iter, lower is better) ===");
    for &(m, k, n) in &[
        (128, 128, 128),
        (512, 512, 512),
        (1, 4096, 4096),
        (32, 128, 4096),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let gpu_ms = bench_matmul(&gpu, dtype, m, k, n, 2, 5)?;
            let cpu_line = match bench_matmul(&cpu, dtype, m, k, n, 1, 3) {
                Ok(cpu_ms) => format!("CPU {cpu_ms:8.2}  ({:.2}x)", cpu_ms / gpu_ms),
                Err(_) => "CPU n/a".to_string(),
            };
            println!("{m:4}x{k:4}x{n:4} {dtype:?}: GPU {gpu_ms:8.2}  {cpu_line}");
        }
    }

    println!("\n=== add (ms/iter) ===");
    for elems in [262_144, 4_194_304] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let gpu_ms = bench_binary(&gpu, dtype, elems, 2, 5)?;
            let cpu_line = match bench_binary(&cpu, dtype, elems, 1, 3) {
                Ok(cpu_ms) => format!("CPU {cpu_ms:8.2}  ({:.2}x)", cpu_ms / gpu_ms),
                Err(_) => "CPU n/a".to_string(),
            };
            println!("{elems:9} {dtype:?}: GPU {gpu_ms:8.2}  {cpu_line}");
        }
    }
    Ok(())
}
