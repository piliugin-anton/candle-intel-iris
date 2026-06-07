use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn bench_gather(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let src = Tensor::rand(-1.0f32, 1.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let idx = Tensor::arange(0u32, K as u32, device)
        .unwrap()
        .reshape((1, 1, K))
        .unwrap()
        .broadcast_as((B, M, K))
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let bytes = B * M * K * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                src.gather(black_box(&idx), 2).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn bench_index_select(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let src = Tensor::rand(-1.0f32, 1.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let idx = Tensor::arange(0u32, M as u32, device)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let bytes = B * M * K * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                src.index_select(black_box(&idx), 1).unwrap();
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
            let dt = format!("{dtype:?}");
            bench_gather(c, &device, dtype, &format!("gather_{dt}"));
            bench_index_select(c, &device, dtype, &format!("index_select_{dt}"));
        }
    }
}

criterion_group!(benches, criterion_benchmark);
