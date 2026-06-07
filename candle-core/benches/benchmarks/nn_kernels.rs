use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{layer_norm, rms_norm, sdpa, softmax_last_dim};
use candle_nn::rotary_emb::rope;
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

const BS: usize = 1;
const HEADS: usize = 8;
const SEQ: usize = 512;
const DK: usize = 64;
const HIDDEN: usize = 768;

fn bench_sdpa(c: &mut Criterion, device: &Device, dtype: DType, causal: bool, name: &str) {
    let scale = (DK as f32).sqrt().recip();
    let q = Tensor::rand(-1.0f32, 1.0f32, (BS, HEADS, SEQ, DK), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k = Tensor::rand(-1.0f32, 1.0f32, (BS, HEADS, SEQ, DK), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let v = Tensor::rand(-1.0f32, 1.0f32, (BS, HEADS, SEQ, DK), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let elems = BS * HEADS * SEQ * DK;
    let bytes = elems * dtype.size_in_bytes() * 3;

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                sdpa(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    None,
                    causal,
                    scale,
                    1.0,
                )
                .unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn bench_softmax(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let xs = Tensor::rand(-1.0f32, 1.0f32, (BS, HEADS, SEQ, SEQ), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bytes = BS * HEADS * SEQ * SEQ * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                softmax_last_dim(black_box(&xs)).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let xs = Tensor::rand(-1.0f32, 1.0f32, (BS, SEQ, HIDDEN), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let alpha = Tensor::ones(HIDDEN, dtype, device).unwrap();
    let bytes = BS * SEQ * HIDDEN * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                rms_norm(black_box(&xs), black_box(&alpha), 1e-5).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn bench_layer_norm(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let xs = Tensor::rand(-1.0f32, 1.0f32, (BS, SEQ, HIDDEN), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let alpha = Tensor::ones(HIDDEN, dtype, device).unwrap();
    let beta = Tensor::zeros(HIDDEN, dtype, device).unwrap();
    let bytes = BS * SEQ * HIDDEN * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                layer_norm(black_box(&xs), black_box(&alpha), black_box(&beta), 1e-5).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn bench_rope(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let emb = DK * 2;
    let xs = Tensor::rand(-1.0f32, 1.0f32, (BS, HEADS, SEQ, emb), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let cos = Tensor::rand(-1.0f32, 1.0f32, (SEQ, emb / 2), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let sin = Tensor::rand(-1.0f32, 1.0f32, (SEQ, emb / 2), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bytes = BS * HEADS * SEQ * emb * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                rope(black_box(&xs), black_box(&cos), black_box(&sin)).unwrap();
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
            bench_sdpa(
                c,
                &device,
                dtype,
                true,
                &format!("sdpa_causal_{dt}"),
            );
            bench_softmax(c, &device, dtype, &format!("softmax_last_dim_{dt}"));
            bench_rms_norm(c, &device, dtype, &format!("rms_norm_{dt}"));
            bench_layer_norm(c, &device, dtype, &format!("layer_norm_{dt}"));
            bench_rope(c, &device, dtype, &format!("rope_{dt}"));
        }
    }
}

criterion_group!(benches, criterion_benchmark);
