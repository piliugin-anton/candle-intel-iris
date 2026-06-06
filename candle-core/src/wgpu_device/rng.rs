use super::error::Result;
use super::kernel::{elemwise_workgroup_count, WgpuKernel};
use super::shader_cache::RANDOM_KERNEL_LAYOUT_KEY;
use super::storage::WgpuStorage;
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::wgsl::RANDOM;
use crate::{DType, Error, Layout, Result as CandleResult, Shape};
use std::sync::{Arc, RwLock};

/// Default RNG seed (matches Metal backend).
pub const DEFAULT_RNG_SEED: u64 = 299_792_458;

/// Uniform block for `random.wgsl` (`RandomParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RandomUniforms {
    pub elem_count: u32,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub param0: f32,
    pub param1: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl RandomUniforms {
    pub fn new(elem_count: usize, seed: u64, param0: f32, param1: f32) -> Self {
        Self {
            elem_count: elem_count as u32,
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            param0,
            param1,
            _pad0: 0,
            _pad1: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is `#[repr(C)]` with no implicit padding beyond `_pad`.
        unsafe {
            std::slice::from_raw_parts((self as *const Self).cast(), std::mem::size_of::<Self>())
        }
    }
}

/// Bind group layout for random kernels (output + params).
#[derive(Clone, Default)]
pub struct RandomBindGroupLayout {
    inner: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
}

impl std::fmt::Debug for RandomBindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomBindGroupLayout")
            .finish_non_exhaustive()
    }
}

impl RandomBindGroupLayout {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    pub fn get_or_create(&self, device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
        {
            let guard = self.inner.read()?;
            if let Some(layout) = guard.as_ref() {
                return Ok(layout.clone());
            }
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("candle random kernel bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mut guard = self.inner.write()?;
        if let Some(existing) = guard.as_ref() {
            return Ok(existing.clone());
        }
        guard.replace(layout.clone());
        Ok(layout)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &wgpu::Buffer,
        uniforms: &RandomUniforms,
    ) -> Result<wgpu::BindGroup> {
        let layout = self.get_or_create(device)?;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle random params"),
            size: std::mem::size_of::<RandomUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, uniforms.as_bytes());

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("candle random bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}

fn compile_random_kernel(device: &WgpuDevice, entry_point: &str) -> Result<WgpuKernel> {
    let bind_layout = device
        .random_bind_group_layout()
        .get_or_create(device.device())?;
    let pipeline = device.shader_cache().get_or_create_pipeline_with_layout(
        device.device(),
        Some(&bind_layout),
        RANDOM_KERNEL_LAYOUT_KEY,
        RANDOM,
        entry_point,
    )?;
    Ok(WgpuKernel::from_random_pipeline(
        pipeline,
        device.caps().elem_workgroup_size,
    ))
}

fn dispatch_random_f32(
    device: &WgpuDevice,
    shape: &Shape,
    entry_point: &str,
    param0: f32,
    param1: f32,
) -> CandleResult<WgpuStorage> {
    let elem_count = shape.elem_count();
    let mut seed = device
        .rng_seed()
        .write()
        .map_err(|e| Error::Msg(e.to_string()))?;
    let uniforms = RandomUniforms::new(elem_count, *seed, param0, param1);
    *seed = seed.wrapping_add(elem_count as u64);

    let out = WgpuStorage::alloc(device, shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(shape);
    let kernel = compile_random_kernel(device, entry_point).map_err(Error::from)?;
    let bind_group = device
        .random_bind_group_layout()
        .create_bind_group(
            device.device(),
            device.queue(),
            out.backing().buffer(),
            &uniforms,
        )
        .map_err(Error::from)?;
    let grid = elemwise_workgroup_count(device, elem_count);
    out.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    let _ = out_layout;
    Ok(out)
}

pub fn dispatch_rand_uniform_f32(
    device: &WgpuDevice,
    shape: &Shape,
    lo: f64,
    up: f64,
) -> CandleResult<WgpuStorage> {
    dispatch_random_f32(device, shape, "rand_uniform_f32", lo as f32, up as f32)
}

pub fn dispatch_rand_normal_f32(
    device: &WgpuDevice,
    shape: &Shape,
    mean: f64,
    stddev: f64,
) -> CandleResult<WgpuStorage> {
    dispatch_random_f32(device, shape, "rand_normal_f32", mean as f32, stddev as f32)
}

pub fn dispatch_rand_uniform(
    device: &WgpuDevice,
    shape: &Shape,
    dtype: DType,
    lo: f64,
    up: f64,
) -> CandleResult<WgpuStorage> {
    match dtype {
        DType::F32 => dispatch_rand_uniform_f32(device, shape, lo, up),
        DType::F16 | DType::BF16 => {
            let f32 = dispatch_rand_uniform_f32(device, shape, lo, up)?;
            let layout = Layout::contiguous(shape);
            f32.to_dtype(&layout, dtype)
        }
        other => Err(Error::UnsupportedDTypeForOp(other, "rand_uniform").bt()),
    }
}

pub fn dispatch_rand_normal(
    device: &WgpuDevice,
    shape: &Shape,
    dtype: DType,
    mean: f64,
    stddev: f64,
) -> CandleResult<WgpuStorage> {
    match dtype {
        DType::F32 => dispatch_rand_normal_f32(device, shape, mean, stddev),
        DType::F16 | DType::BF16 => {
            let f32 = dispatch_rand_normal_f32(device, shape, mean, stddev)?;
            let layout = Layout::contiguous(shape);
            f32.to_dtype(&layout, dtype)
        }
        other => Err(Error::UnsupportedDTypeForOp(other, "rand_normal").bt()),
    }
}
