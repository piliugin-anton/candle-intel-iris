use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

const BATCH: usize = 4;
const SEQ: usize = 128;
const HIDDEN: usize = 64;
const N: usize = 8;

fn run_cat(tensors: &[Tensor], dim: usize) {
    Tensor::cat(tensors, dim).unwrap();
}

fn make_u8_contiguous(device: &Device) -> [Tensor; N] {
    std::array::from_fn(|_| {
        Tensor::zeros((BATCH, SEQ, HIDDEN), DType::U8, device).unwrap()
    })
}

fn bench_copy2d_u8(c: &mut Criterion, device: &Device) {
    let total_bytes = (N * BATCH * SEQ * HIDDEN) as u64;

    {
        let tensors = make_u8_contiguous(device);
        let mut group = c.benchmark_group(device.bench_name("copy2d_u8_cat_contig_dim1"));
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_function("iter", |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    run_cat(black_box(&tensors), 1);
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
        group.finish();
    }

    {
        let tensors = make_u8_contiguous(device);
        let mut group = c.benchmark_group(device.bench_name("copy2d_u8_cat_contig_dim0"));
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_function("iter", |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    run_cat(black_box(&tensors), 0);
                }
                device.sync().unwrap();
                start.elapsed()
            })
        });
        group.finish();
    }

}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in &handler.devices {
        bench_copy2d_u8(c, device);
    }
}

criterion_group!(benches, criterion_benchmark);
