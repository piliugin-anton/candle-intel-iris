mod adapter;
mod allocator;
mod async_io;
mod bind_group;
mod error;
mod intel_caps;
mod kernel;
mod mapped_buffer;
mod ops;
mod shader_cache;
mod storage;

pub use crate::wgsl::{
    BINARY, CAST, COMMON, COPY, ELEM_WORKGROUP_SIZE, MATMUL_NAIVE, MATMUL_TILED,
    MATMUL_WORKGROUP_SIZE, REDUCE, REDUCE_WORKGROUP_SIZE, UNARY,
};
pub use adapter::{is_intel_adapter, WgpuDeviceConfig, INTEL_VENDOR_ID};
pub use allocator::Allocator;
pub use bind_group::{
    BindGroupBuilder, KernelUniforms, StandardBindGroupArgs, StandardBindGroupLayout,
    TensorLayoutUniform, MAX_TENSOR_DIMS,
};
pub use error::{Result, WgpuError};
pub use intel_caps::{
    detect_generation, inject_workgroup_size, tune_shader_source, IntelCaps, IntelGeneration,
    DEFAULT_UMA_AUTO_MAP_THRESHOLD,
};
pub use kernel::{KernelDispatchArgs, WgpuKernel};
pub use mapped_buffer::{MappedBacking, MAPPED_READ_USAGE};
pub use shader_cache::{ShaderCache, STANDARD_KERNEL_LAYOUT_KEY};
pub use storage::{buffer_offset, BufferBacking, BufferOffset, WgpuStorage, STORAGE_BUFFER_USAGE};

use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Error, Result as CandleResult, Shape};
use std::sync::atomic::{AtomicUsize, Ordering};

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
    shader_cache: ShaderCache,
    allocator: Allocator,
}

impl WgpuDevice {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
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
            shader_cache: ShaderCache::new(),
            allocator: Allocator::new(),
        }
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn caps(&self) -> &IntelCaps {
        &self.caps
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
        self.storage_from_cpu_storage(&T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> CandleResult<Self::Storage> {
        WgpuStorage::from_cpu(self, storage).map_err(Error::from)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> CandleResult<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> CandleResult<Self::Storage> {
        Err(Error::Msg(
            "wgpu backend: rand_uniform not yet implemented".into(),
        ))
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> CandleResult<Self::Storage> {
        Err(Error::Msg(
            "wgpu backend: rand_normal not yet implemented".into(),
        ))
    }

    fn set_seed(&self, _: u64) -> CandleResult<()> {
        Err(Error::Msg(
            "wgpu backend: set_seed not yet implemented".into(),
        ))
    }

    fn get_current_seed(&self) -> CandleResult<u64> {
        Err(Error::Msg(
            "wgpu backend: get_current_seed not yet implemented".into(),
        ))
    }

    fn synchronize(&self) -> CandleResult<()> {
        async_io::poll_device(self.device()).map_err(Error::from)
    }
}
