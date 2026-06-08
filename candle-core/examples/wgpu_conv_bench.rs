//! CPU vs WGPU timing comparison for conv1d / conv2d.
//!
//! ```text
//! cargo run -p candle-core --features wgpu --release --example wgpu_conv_bench
//! ```

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const WARMUP: u32 = 2;
const ITERS: u32 = 10;

struct Conv2dCase {
    label: &'static str,
    input: (usize, usize, usize, usize),
    kernel: (usize, usize, usize, usize),
    padding: usize,
    stride: usize,
    dilation: usize,
}

struct Conv1dCase {
    label: &'static str,
    input: (usize, usize, usize),
    kernel: (usize, usize, usize),
    padding: usize,
    stride: usize,
    dilation: usize,
}

fn bench_conv2d(
    device: &Device,
    dtype: DType,
    case: &Conv2dCase,
) -> candle_core::Result<f64> {
    let input = Tensor::randn(0f32, 1.0, case.input, device)?.to_dtype(dtype)?;
    let kernel = Tensor::randn(0f32, 1.0, case.kernel, device)?.to_dtype(dtype)?;
    for _ in 0..WARMUP {
        let _ = input.conv2d(&kernel, case.padding, case.stride, case.dilation, 1)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = input.conv2d(&kernel, case.padding, case.stride, case.dilation, 1)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERS))
}

fn bench_conv1d(
    device: &Device,
    dtype: DType,
    case: &Conv1dCase,
) -> candle_core::Result<f64> {
    let input = Tensor::randn(0f32, 1.0, case.input, device)?.to_dtype(dtype)?;
    let kernel = Tensor::randn(0f32, 1.0, case.kernel, device)?.to_dtype(dtype)?;
    for _ in 0..WARMUP {
        let _ = input.conv1d(&kernel, case.padding, case.stride, case.dilation, 1)?;
        device.synchronize()?;
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = input.conv1d(&kernel, case.padding, case.stride, case.dilation, 1)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERS))
}

fn print_conv2d_row(
    cpu: &Device,
    gpu: &Device,
    dtype: DType,
    case: &Conv2dCase,
) -> candle_core::Result<()> {
    let gpu_ms = bench_conv2d(gpu, dtype, case)?;
    let cpu_line = match bench_conv2d(cpu, dtype, case) {
        Ok(cpu_ms) => format!("CPU {cpu_ms:8.2}  (GPU {:.2}x)", cpu_ms / gpu_ms),
        Err(_) => "CPU n/a".to_string(),
    };
    println!(
        "{:24} {dtype:?}: GPU {gpu_ms:8.2}  {cpu_line}",
        case.label
    );
    Ok(())
}

fn print_conv1d_row(
    cpu: &Device,
    gpu: &Device,
    dtype: DType,
    case: &Conv1dCase,
) -> candle_core::Result<()> {
    let gpu_ms = bench_conv1d(gpu, dtype, case)?;
    let cpu_line = match bench_conv1d(cpu, dtype, case) {
        Ok(cpu_ms) => format!("CPU {cpu_ms:8.2}  (GPU {:.2}x)", cpu_ms / gpu_ms),
        Err(_) => "CPU n/a".to_string(),
    };
    println!(
        "{:24} {dtype:?}: GPU {gpu_ms:8.2}  {cpu_line}",
        case.label
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_wgpu()?;
    println!("WGPU device: {gpu:?}\n");

    let conv2d_cases = [
        Conv2dCase {
            label: "resnet_block_56x56",
            input: (1, 4, 56, 56),
            kernel: (8, 4, 3, 3),
            padding: 1,
            stride: 1,
            dilation: 1,
        },
        Conv2dCase {
            label: "small_8x8",
            input: (1, 3, 8, 8),
            kernel: (4, 3, 3, 3),
            padding: 1,
            stride: 1,
            dilation: 1,
        },
        Conv2dCase {
            label: "mnist_28x28",
            input: (32, 1, 28, 28),
            kernel: (32, 1, 5, 5),
            padding: 0,
            stride: 1,
            dilation: 1,
        },
        Conv2dCase {
            label: "wide_128x128",
            input: (1, 64, 128, 128),
            kernel: (128, 64, 3, 3),
            padding: 1,
            stride: 1,
            dilation: 1,
        },
    ];

    let conv1d_cases = [
        Conv1dCase {
            label: "audio_16",
            input: (2, 4, 16),
            kernel: (3, 4, 3),
            padding: 0,
            stride: 1,
            dilation: 1,
        },
        Conv1dCase {
            label: "speech_512",
            input: (8, 64, 512),
            kernel: (128, 64, 7),
            padding: 3,
            stride: 1,
            dilation: 1,
        },
    ];

    println!("=== conv2d (ms/iter; ratio = CPU/GPU, >1 means GPU faster) ===");
    for case in &conv2d_cases {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            print_conv2d_row(&cpu, &gpu, dtype, case)?;
        }
    }

    println!("\n=== conv1d (ms/iter) ===");
    for case in &conv1d_cases {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            print_conv1d_row(&cpu, &gpu, dtype, case)?;
        }
    }

    Ok(())
}
