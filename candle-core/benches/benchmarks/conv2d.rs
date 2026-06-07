use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn bench_conv2d(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let input = Tensor::rand(-1.0f32, 1.0f32, (1, 4, 56, 56), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let kernel = Tensor::rand(-1.0f32, 1.0f32, (8, 4, 3, 3), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bytes = input.elem_count() * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(input.conv2d(black_box(&kernel), 1, 1, 1, 1)).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            bench_conv2d(c, &device, dtype, &format!("conv2d_{dtype:?}"));
        }
    }
}

criterion_group!(benches, criterion_benchmark);
