// GPU dispatch helpers pass tensor buffers, layouts, and uniforms (CUDA-parity arity).
#![allow(clippy::too_many_arguments)]

mod adapter;
mod allocator;
mod async_io;
mod bind_group;
mod conv;
mod error;
mod fallback;
mod fill;
mod indexing;
mod intel_caps;
mod kernel;
mod mapped_buffer;
mod ops;
mod rng;
mod shader_cache;
mod sort;
mod storage;

pub use crate::wgsl::{
    BINARY, BINARY_BF16, CAST, COMMON, COPY, COPY2D, ELEM_WORKGROUP_SIZE, MATMUL_NAIVE,
    MATMUL_TILED, MATMUL_TILED_BF16, MATMUL_TILED_F16, MATMUL_TILED_VEC, MATMUL_VEC_WIDTH,
    MATMUL_WORKGROUP_SIZE, QMATMUL_Q4_0, QMATMUL_Q4_K, QMATMUL_Q5_0, QMATMUL_Q8_0, REDUCE,
    REDUCE_WORKGROUP_SIZE, RMS_NORM, ROPE, SDPA_FULL, SDPA_VECTOR, SOFTMAX, UNARY, UNARY_BF16,
    UNARY_F16, WHERE_COND,
};
pub use adapter::{is_intel_adapter, WgpuDeviceConfig, INTEL_VENDOR_ID};
pub use allocator::Allocator;
pub use async_io::wait_for_buffer_map;
pub use bind_group::{
    BindGroupBuilder, ExtendedBindGroupArgs, ExtendedBindGroupBuilder, ExtendedBindGroupLayout,
    KernelUniforms, StandardBindGroupArgs, StandardBindGroupLayout, TensorLayoutUniform,
    MAX_TENSOR_DIMS,
};
pub use error::{Result, WgpuError};
pub use fallback::{
    cpu_fallback_inplace_op1, cpu_fallback_inplace_op2, cpu_fallback_inplace_op3, cpu_fallback_op1,
    cpu_fallback_op2, cpu_fallback_op3,
};
pub use intel_caps::{
    detect_generation, inject_workgroup_size, tune_matmul_shader_source, tune_shader_source,
    IntelCaps, IntelGeneration, DEFAULT_UMA_AUTO_MAP_THRESHOLD,
};
pub use kernel::{KernelDispatchArgs, WgpuKernel};
pub use mapped_buffer::{MappedBacking, MAPPED_READ_USAGE};
pub use ops::{
    dispatch_copy2d, dispatch_copy_strided_src, dispatch_dequant_f32, dispatch_layer_norm,
    dispatch_layer_norm_f32, dispatch_qmatmul_q4_0, dispatch_qmatmul_q4_k, dispatch_qmatmul_q5_0,
    dispatch_qmatmul_q8_0, dispatch_quant_f32, dispatch_rms_norm, dispatch_rms_norm_f32,
    dispatch_rope, dispatch_rope_f32, dispatch_rope_i, dispatch_rope_thd, dispatch_sdpa,
    dispatch_sdpa_f32, dispatch_sdpa_full, dispatch_sdpa_full_f32, dispatch_sdpa_vector,
    dispatch_sdpa_vector_f32, dispatch_sigmoid, dispatch_softmax_last_dim,
    dispatch_softmax_last_dim_f32, dispatch_where_u8_f32, gpu_dequant_supported,
    gpu_quant_supported, upload_q4_0_weights, upload_quant_weights, Copy2dParams, MAX_SDPA_DIM,
};
pub use shader_cache::{ShaderCache, STANDARD_KERNEL_LAYOUT_KEY};
pub use sort::{dispatch_arg_sort_last_dim, gpu_argsort_supported, MAX_ARGSORT_NCOLS_PAD};
pub use storage::{buffer_offset, BufferBacking, BufferOffset, WgpuStorage, STORAGE_BUFFER_USAGE};
pub(crate) use storage::read_device_buffer_region;

use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Error, Result as CandleResult, Shape};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Unique identifier for wgpu device handles (distinct from the PCI device id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// wgpu device handle with shader and buffer caching.
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    id: DeviceId,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    caps: IntelCaps,
    storage_buffer_offset_alignment: u32,
    shader_cache: ShaderCache,
    allocator: Allocator,
    rng_seed: Arc<RwLock<u64>>,
    random_layout: rng::RandomBindGroupLayout,
}

impl WgpuDevice {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let storage_buffer_offset_alignment =
            device.limits().min_storage_buffer_offset_alignment;
        Self {
            id: DeviceId::new(),
            device,
            queue,
            adapter_info: wgpu::AdapterInfo {
                name: String::new(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Other,
                device_pci_bus_id: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Noop,
                subgroup_min_size: 0,
                subgroup_max_size: 0,
                transient_saves_memory: false,
            },
            caps: IntelCaps::default_fallback(),
            storage_buffer_offset_alignment,
            shader_cache: ShaderCache::new(),
            allocator: Allocator::new(),
            rng_seed: Arc::new(RwLock::new(rng::DEFAULT_RNG_SEED)),
            random_layout: rng::RandomBindGroupLayout::new(),
        }
    }

    pub(crate) fn rng_seed(&self) -> &Arc<RwLock<u64>> {
        &self.rng_seed
    }

    pub(crate) fn random_bind_group_layout(&self) -> &rng::RandomBindGroupLayout {
        &self.random_layout
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn caps(&self) -> &IntelCaps {
        &self.caps
    }

    /// Minimum byte alignment required for storage-buffer bind offsets on this device.
    pub fn storage_buffer_offset_alignment(&self) -> u32 {
        self.storage_buffer_offset_alignment
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn shader_cache(&self) -> &ShaderCache {
        &self.shader_cache
    }

    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }

    /// Dtype used for GPU compute after Intel generation policy is applied.
    pub fn effective_dtype(&self, requested: DType) -> DType {
        self.caps.effective_compute_dtype(requested)
    }

    pub fn allocate_buffer(
        &self,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> Result<std::sync::Arc<wgpu::Buffer>> {
        self.allocator.allocate(&self.device, size, usage)
    }

    pub fn get_compute_pipeline(
        &self,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline> {
        self.shader_cache
            .get_or_create_pipeline(&self.device, source, entry_point)
    }

    pub fn drop_unused_buffers(&self) -> Result<()> {
        self.allocator.drop_unused()
    }

    /// Allocate storage that is always UMA-mapped regardless of size heuristics.
    pub fn alloc_pinned_mapped(&self, shape: &Shape, dtype: DType) -> Result<WgpuStorage> {
        WgpuStorage::alloc_mapped(self, shape, dtype, true)
    }

    pub fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu {
            gpu_id: self.adapter_info.device as usize,
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    #[cfg(test)]
    pub(crate) fn new_test(integrated: bool, uma_threshold: usize) -> Self {
        let (wgpu_device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        });
        let mut device = Self::new(wgpu_device, queue);
        device.adapter_info = wgpu::AdapterInfo {
            name: "Intel Iris Xe".into(),
            vendor: INTEL_VENDOR_ID,
            device: 0x9A49,
            device_type: if integrated {
                wgpu::DeviceType::IntegratedGpu
            } else {
                wgpu::DeviceType::DiscreteGpu
            },
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Noop,
            subgroup_min_size: 8,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        };
        device.caps = IntelCaps {
            uma_auto_map_threshold: uma_threshold,
            is_integrated: integrated,
            elem_workgroup_size: 8,
            reduce_workgroup_size: 8,
            ..IntelCaps::default_fallback()
        };
        device
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_ordinal: usize) -> CandleResult<Self> {
        Self::new_default().map_err(|e| Error::Msg(e.to_string()))
    }

    fn location(&self) -> crate::DeviceLocation {
        self.location()
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.same_device(rhs)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> CandleResult<Self::Storage> {
        let storage = WgpuStorage::alloc(self, shape, dtype)?;
        let zeros = match dtype {
            DType::U8 => CpuStorage::U8(vec![0; shape.elem_count()]),
            DType::U32 => CpuStorage::U32(vec![0; shape.elem_count()]),
            DType::I16 => CpuStorage::I16(vec![0; shape.elem_count()]),
            DType::I32 => CpuStorage::I32(vec![0; shape.elem_count()]),
            DType::I64 => CpuStorage::I64(vec![0; shape.elem_count()]),
            DType::BF16 => CpuStorage::BF16(vec![half::bf16::from_f32(0.); shape.elem_count()]),
            DType::F16 => CpuStorage::F16(vec![half::f16::from_f32(0.); shape.elem_count()]),
            DType::F32 => CpuStorage::F32(vec![0.; shape.elem_count()]),
            DType::F64 => CpuStorage::F64(vec![0.; shape.elem_count()]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![float8::F8E4M3::ZERO; shape.elem_count()]),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(dtype, "zeros").bt());
            }
        };
        let (bytes, _) = storage::cpu_storage_as_bytes(&zeros).map_err(Error::from)?;
        storage.write_bytes(bytes).map_err(Error::from)?;
        Ok(storage)
    }

    // SAFETY: Delegates to `WgpuStorage::alloc`, which zero-initializes GPU buffers; no
    // uninitialized host memory is exposed to callers.
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> CandleResult<Self::Storage> {
        WgpuStorage::alloc(self, shape, dtype).map_err(Error::from)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> CandleResult<Self::Storage> {
        let bytes = storage::typed_slice_as_bytes(s);
        WgpuStorage::from_bytes(self, bytes, s.len(), T::DTYPE).map_err(Error::from)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> CandleResult<Self::Storage> {
        WgpuStorage::from_cpu(self, storage).map_err(Error::from)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> CandleResult<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> CandleResult<Self::Storage> {
        rng::dispatch_rand_uniform(self, shape, dtype, lo, up)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> CandleResult<Self::Storage> {
        rng::dispatch_rand_normal(self, shape, dtype, mean, stddev)
    }

    fn set_seed(&self, seed: u64) -> CandleResult<()> {
        *self
            .rng_seed
            .write()
            .map_err(|e| Error::Msg(e.to_string()))? = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> CandleResult<u64> {
        Ok(*self
            .rng_seed
            .read()
            .map_err(|e| Error::Msg(e.to_string()))?)
    }

    fn synchronize(&self) -> CandleResult<()> {
        async_io::poll_device(self.device()).map_err(Error::from)
    }
}
