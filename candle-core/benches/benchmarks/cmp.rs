use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn bench_lt(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let lhs = Tensor::rand(-1.0f32, 1.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let rhs = Tensor::rand(-1.0f32, 1.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bytes = B * M * K * (dtype.size_in_bytes() * 2 + 1);

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                lhs.lt(black_box(&rhs)).unwrap();
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
            bench_lt(c, &device, dtype, &format!("cmp_lt_{dtype:?}"));
        }
    }
}

criterion_group!(benches, criterion_benchmark);
