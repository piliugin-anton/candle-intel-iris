//! Simple WGPU workload for manual profiling (Intel GPA, Tracy, etc.).
//!
//! ```text
//! cargo run -p candle-core --features wgpu --release --example wgpu_bench
//! ```

use candle_core::{Device, Result, Tensor};
use std::time::Instant;

const WARMUP: u32 = 3;
const ITERS: u32 = 20;
const MATMUL_N: usize = 512;
// Elem-wise kernels use one workgroup per element (max dispatch.x = 65535).
const ADD_ELEMS: usize = 65_535;

fn linspace_f32(n: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-4 - 0.5).collect();
    Tensor::from_vec(data, (n,), device)
}

fn bench_matmul(device: &Device) -> Result<f64> {
    let n = MATMUL_N * MATMUL_N;
    let a = linspace_f32(n, device)?.reshape((MATMUL_N, MATMUL_N))?;
    let b = linspace_f32(n, device)?.reshape((MATMUL_N, MATMUL_N))?;
    for _ in 0..WARMUP {
        let _ = a.matmul(&b)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = a.matmul(&b)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERS))
}

fn bench_add(device: &Device) -> Result<f64> {
    let a = linspace_f32(ADD_ELEMS, device)?;
    let b = linspace_f32(ADD_ELEMS, device)?;
    for _ in 0..WARMUP {
        let _ = a.add(&b)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = a.add(&b)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERS))
}

fn main() -> Result<()> {
    let device = Device::new_wgpu()?;
    println!("WGPU bench on {device:?}");
    println!(
        "matmul ({MATMUL_N}x{MATMUL_N}): {:.2} ms/iter",
        bench_matmul(&device)?
    );
    println!(
        "add ({ADD_ELEMS} elems): {:.2} ms/iter",
        bench_add(&device)?
    );
    Ok(())
}
