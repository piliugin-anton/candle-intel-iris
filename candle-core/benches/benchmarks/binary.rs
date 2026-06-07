use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run_binary_benchmark<F>(c: &mut Criterion, device: &Device, dtype: DType, name: &str, op: F)
where
    F: Fn(&Tensor, &Tensor) + Copy,
{
    let b = 1;
    let m = 1024;
    let k = 1024;

    let lhs = Tensor::arange(0.0f32, (b * m * k) as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((b, m, k))
        .unwrap();

    let rhs = Tensor::arange(0.0f32, (b * m * k) as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((b, m, k))
        .unwrap();

    let flops = 2 * b * m * k * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                op(black_box(&lhs), black_box(&rhs));
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
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let dt = format!("{dtype:?}");
            run_binary_benchmark(c, &device, dtype, &format!("binary_add_{dt}"), |l, r| {
                l.add(r).unwrap();
            });
            run_binary_benchmark(c, &device, dtype, &format!("binary_mul_{dt}"), |l, r| {
                l.mul(r).unwrap();
            });
            run_binary_benchmark(c, &device, dtype, &format!("binary_sub_{dt}"), |l, r| {
                l.sub(r).unwrap();
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
