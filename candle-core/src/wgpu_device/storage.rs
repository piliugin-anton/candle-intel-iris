use super::async_io::wait_for_buffer_map;
use super::error::{Result, WgpuError};
use super::kernel::WgpuKernel;
use super::mapped_buffer::MappedBacking;
use super::ops;
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::wgsl::CAST;
use crate::{CpuStorage, DType, Error, Layout, Result as CandleResult, Shape};
use std::sync::Arc;
use wgpu::BufferUsages;

/// Usages for tensor storage buffers used by compute kernels.
pub const STORAGE_BUFFER_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// How tensor data is backed on the GPU.
#[derive(Debug, Clone)]
pub enum BufferBacking {
    DeviceLocal(Arc<wgpu::Buffer>),
    Mapped(MappedBacking),
}

impl BufferBacking {
    pub fn buffer(&self) -> &Arc<wgpu::Buffer> {
        match self {
            Self::DeviceLocal(buf) => buf,
            Self::Mapped(m) => m.storage(),
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }

    pub fn with_unmapped<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        match self {
            Self::DeviceLocal(_) => f(),
            Self::Mapped(m) => m.with_unmapped(f),
        }
    }
}

/// Binding view into a wgpu buffer at the byte offset implied by a Candle [`Layout`].
#[derive(Debug, Clone)]
pub struct BufferOffset<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset_in_bytes: u64,
}

pub fn buffer_offset<'a>(storage: &'a WgpuStorage, layout: &Layout) -> BufferOffset<'a> {
    BufferOffset {
        buffer: storage.backing.buffer(),
        offset_in_bytes: (layout.start_offset() * storage.dtype.size_in_bytes()) as u64,
    }
}

/// GPU tensor storage: a typed wgpu buffer owned by a [`WgpuDevice`].
#[derive(Debug, Clone)]
pub struct WgpuStorage {
    backing: BufferBacking,
    device: WgpuDevice,
    elem_count: usize,
    dtype: DType,
}

impl WgpuStorage {
    pub fn new(
        backing: BufferBacking,
        device: WgpuDevice,
        elem_count: usize,
        dtype: DType,
    ) -> Self {
        Self {
            backing,
            device,
            elem_count,
            dtype,
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn backing(&self) -> &BufferBacking {
        &self.backing
    }

    pub fn buffer(&self) -> &Arc<wgpu::Buffer> {
        self.backing.buffer()
    }

    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    pub fn is_mapped(&self) -> bool {
        self.backing.is_mapped()
    }

    fn byte_len(&self) -> usize {
        self.elem_count * self.dtype.size_in_bytes()
    }

    fn should_use_mapped(device: &WgpuDevice, byte_len: usize, force_mapped: bool) -> bool {
        force_mapped
            || device
                .caps()
                .should_auto_map(device.adapter_info(), byte_len)
    }

    fn alloc_backing(
        device: &WgpuDevice,
        byte_len: usize,
        force_mapped: bool,
    ) -> Result<BufferBacking> {
        if Self::should_use_mapped(device, byte_len, force_mapped) {
            Ok(BufferBacking::Mapped(MappedBacking::new(device, byte_len)?))
        } else {
            let buffer = device.allocate_buffer(byte_len, STORAGE_BUFFER_USAGE)?;
            Ok(BufferBacking::DeviceLocal(buffer))
        }
    }

    /// Allocates a new uninitialized storage buffer on `device`.
    pub fn alloc(device: &WgpuDevice, shape: &Shape, dtype: DType) -> Result<Self> {
        Self::alloc_mapped(device, shape, dtype, false)
    }

    /// Allocates storage, optionally forcing UMA-mapped backing.
    pub fn alloc_mapped(
        device: &WgpuDevice,
        shape: &Shape,
        dtype: DType,
        force_mapped: bool,
    ) -> Result<Self> {
        let elem_count = shape.elem_count();
        let byte_len = elem_count * dtype.size_in_bytes();
        let backing = Self::alloc_backing(device, byte_len, force_mapped)?;
        Ok(Self::new(backing, device.clone(), elem_count, dtype))
    }

    /// Uploads CPU storage into a freshly allocated GPU buffer.
    pub fn from_cpu(device: &WgpuDevice, storage: &CpuStorage) -> Result<Self> {
        let dtype = storage.dtype();
        let (bytes, elem_count) = cpu_storage_as_bytes(storage)?;
        let byte_len = bytes.len();
        let use_mapped = Self::should_use_mapped(device, byte_len, false);
        let backing = if use_mapped {
            let mapped = MappedBacking::new(device, byte_len)?;
            device.queue().write_buffer(mapped.storage(), 0, bytes);
            BufferBacking::Mapped(mapped)
        } else {
            let buffer = device.allocate_buffer(byte_len, STORAGE_BUFFER_USAGE)?;
            device.queue().write_buffer(&buffer, 0, bytes);
            BufferBacking::DeviceLocal(buffer)
        };
        Ok(Self::new(backing, device.clone(), elem_count, dtype))
    }

    pub(crate) fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        match &self.backing {
            BufferBacking::Mapped(m) => {
                self.device.queue().write_buffer(m.storage(), 0, bytes);
                Ok(())
            }
            BufferBacking::DeviceLocal(buf) => {
                self.device.queue().write_buffer(buf, 0, bytes);
                Ok(())
            }
        }
    }

    fn read_bytes(&self) -> Result<Vec<u8>> {
        let size = self.byte_len() as u64;
        match &self.backing {
            BufferBacking::Mapped(m) => m.read_bytes(&self.device, size),
            BufferBacking::DeviceLocal(buf) => read_bytes_staging(self.device(), buf, size),
        }
    }

    fn to_cpu_typed<T: Clone>(&self) -> Result<Vec<T>> {
        let bytes = self.read_bytes()?;
        let elem_size = std::mem::size_of::<T>();
        if bytes.len() % elem_size != 0 {
            return Err(WgpuError::Message(format!(
                "readback size {} is not a multiple of element size {elem_size}",
                bytes.len()
            )));
        }
        let count = bytes.len() / elem_size;
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(elem_size) {
            // SAFETY: `chunk` is exactly `elem_size` bytes and aligned for `T` because the
            // buffer length was validated as a multiple of `size_of::<T>()`.
            let value = unsafe { std::ptr::read(chunk.as_ptr().cast::<T>()) };
            out.push(value);
        }
        Ok(out)
    }

    fn cast_via_cpu(&self, _layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu = self
            .to_cpu_storage()
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        let casted = cast_cpu_storage(&cpu, dtype)?;
        Self::from_cpu(&self.device, &casted)
    }

    fn cast_via_gpu(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let entry = cast_entry_point(self.dtype, dtype).ok_or_else(|| {
            WgpuError::Message(format!(
                "wgpu cast {:?} -> {:?} not implemented",
                self.dtype, dtype
            ))
        })?;

        let out_shape = layout.shape();
        let out = Self::alloc(&self.device, out_shape, dtype)?;
        let out_layout = Layout::contiguous(out_shape);
        let kernel = WgpuKernel::compile_with_workgroup_size(&self.device, CAST, entry, 1)?;
        let uniforms = super::bind_group::KernelUniforms::new(
            out_layout.shape().elem_count(),
            &out_layout,
            layout,
            None,
        );
        let bind_group_builder = super::bind_group::BindGroupBuilder::new();
        let bind_group = bind_group_builder.create_bind_group_bytes(
            self.device.device(),
            self.device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(self, layout),
            None,
            uniforms.as_bytes(),
        )?;
        // Cast kernels loop inside a single workgroup (packed f16/bf16/u8 writes are not race-safe).
        self.backing
            .with_unmapped(|| kernel.dispatch_bind_group(&self.device, &bind_group, [1, 1, 1]))?;
        Ok(out)
    }
}

fn read_bytes_staging(device: &WgpuDevice, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    let wgpu_device = device.device();
    let queue = device.queue();

    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu readback staging"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu readback"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    wait_for_buffer_map(wgpu_device, &rx)?;

    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(bytes)
}

fn cast_entry_point(from: DType, to: DType) -> Option<&'static str> {
    match (from, to) {
        (DType::F32, DType::F16) => Some("cast_f32_f16"),
        (DType::F16, DType::F32) => Some("cast_f16_f32"),
        (DType::F32, DType::BF16) => Some("cast_f32_bf16"),
        (DType::BF16, DType::F32) => Some("cast_bf16_f32"),
        (DType::F16, DType::BF16) => Some("cast_f16_bf16"),
        (DType::BF16, DType::F16) => Some("cast_bf16_f16"),
        (DType::U8, DType::F32) => Some("cast_u8_f32"),
        (DType::F32, DType::U8) => Some("cast_f32_u8"),
        _ => None,
    }
}

/// Maps a typed slice to a new `Vec` during host-side dtype casts.
///
/// # Examples
///
/// ```ignore
/// cast_vec!(data, f16, |v| f16::from_f32(*v))
/// // => Vec<f16>
/// ```
macro_rules! cast_vec {
    ($data:expr, $ty:ty, $map:expr) => {
        $data.iter().map($map).collect::<Vec<$ty>>()
    };
}

fn cast_cpu_storage(storage: &CpuStorage, dtype: DType) -> Result<CpuStorage> {
    use half::{bf16, f16};
    Ok(match (storage, dtype) {
        (CpuStorage::F32(data), DType::F16) => {
            CpuStorage::F16(cast_vec!(data, f16, |v| f16::from_f32(*v)))
        }
        (CpuStorage::F16(data), DType::F32) => {
            CpuStorage::F32(cast_vec!(data, f32, |v| v.to_f32()))
        }
        (CpuStorage::F32(data), DType::BF16) => {
            CpuStorage::BF16(cast_vec!(data, bf16, |v| bf16::from_f32(*v)))
        }
        (CpuStorage::BF16(data), DType::F32) => {
            CpuStorage::F32(cast_vec!(data, f32, |v| v.to_f32()))
        }
        (s, _) if s.dtype() == dtype => s.clone(),
        (s, dt) => {
            return Err(WgpuError::Message(format!(
                "cpu cast {:?} -> {dt:?} not implemented",
                s.dtype()
            )));
        }
    })
}

pub(crate) fn cpu_storage_as_bytes(storage: &CpuStorage) -> Result<(&[u8], usize)> {
    match storage {
        CpuStorage::U8(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::U32(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::I16(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::I32(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::I64(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::BF16(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::F16(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::F32(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::F64(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::F8E4M3(data) => Ok((bytemuck_slice(data), data.len())),
        CpuStorage::F6E2M3(_)
        | CpuStorage::F6E3M2(_)
        | CpuStorage::F4(_)
        | CpuStorage::F8E8M0(_) => Err(WgpuError::Message(format!(
            "unsupported dtype {:?} for wgpu upload",
            storage.dtype()
        ))),
    }
}

fn bytemuck_slice<T>(data: &[T]) -> &[u8] {
    // SAFETY: Reinterpreting a contiguous slice of plain-old-data elements as bytes is
    // valid for the numeric tensor dtypes uploaded to wgpu.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

/// Returns a Candle `Error::Msg` for an unimplemented wgpu backend op.
///
/// # Examples
///
/// ```ignore
/// wgpu_not_impl!("conv1d")
/// // => Err(Error::Msg("wgpu backend: conv1d not yet implemented"))
/// ```
macro_rules! wgpu_not_impl {
    ($name:expr) => {
        Err(Error::Msg(format!(
            "wgpu backend: {} not yet implemented",
            $name
        )))
    };
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> CandleResult<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> CandleResult<CpuStorage> {
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(self.to_cpu_typed()?),
            DType::U32 => CpuStorage::U32(self.to_cpu_typed()?),
            DType::I16 => CpuStorage::I16(self.to_cpu_typed()?),
            DType::I32 => CpuStorage::I32(self.to_cpu_typed()?),
            DType::I64 => CpuStorage::I64(self.to_cpu_typed()?),
            DType::BF16 => CpuStorage::BF16(self.to_cpu_typed()?),
            DType::F16 => CpuStorage::F16(self.to_cpu_typed()?),
            DType::F32 => CpuStorage::F32(self.to_cpu_typed()?),
            DType::F64 => CpuStorage::F64(self.to_cpu_typed()?),
            DType::F8E4M3 => CpuStorage::F8E4M3(self.to_cpu_typed()?),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(Error::UnsupportedDTypeForOp(self.dtype, "to_cpu_storage").bt());
            }
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> CandleResult<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        if cast_entry_point(self.dtype, dtype).is_some() {
            return self.cast_via_gpu(layout, dtype).map_err(Error::from);
        }
        self.cast_via_cpu(layout, dtype).map_err(Error::from)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> CandleResult<Self> {
        ops::dispatch_affine(self, layout, mul, add)
    }

    fn powf(&self, _: &Layout, _: f64) -> CandleResult<Self> {
        wgpu_not_impl!("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> CandleResult<Self> {
        wgpu_not_impl!("elu")
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> CandleResult<Self> {
        ops::dispatch_reduce(self, op, layout, reduce_dims)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> CandleResult<Self> {
        wgpu_not_impl!("cmp")
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> CandleResult<Self> {
        ops::dispatch_unary::<B>(self, layout)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> CandleResult<Self> {
        ops::dispatch_binary::<B>(self, rhs, lhs_layout, rhs_layout)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        on_true: &Self,
        on_true_layout: &Layout,
        on_false: &Self,
        on_false_layout: &Layout,
    ) -> CandleResult<Self> {
        ops::dispatch_where_u8_f32(
            self,
            on_true,
            on_false,
            layout,
            on_true_layout,
            on_false_layout,
        )
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("conv_transpose1d")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("conv2d")
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("conv_transpose2d")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> CandleResult<Self> {
        wgpu_not_impl!("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> CandleResult<Self> {
        wgpu_not_impl!("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> CandleResult<Self> {
        wgpu_not_impl!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> CandleResult<Self> {
        wgpu_not_impl!("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("upsample_bilinear2d")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> CandleResult<Self> {
        wgpu_not_impl!("gather")
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> CandleResult<()> {
        wgpu_not_impl!("scatter_set")
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> CandleResult<()> {
        wgpu_not_impl!("scatter_add_set")
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> CandleResult<Self> {
        wgpu_not_impl!("index_select")
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> CandleResult<Self> {
        wgpu_not_impl!("index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> CandleResult<Self> {
        ops::dispatch_matmul(self, rhs, bmnk, lhs_layout, rhs_layout)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> CandleResult<()> {
        ops::dispatch_copy_strided_src(self, dst, dst_offset, src_l).map_err(Error::from)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> CandleResult<()> {
        ops::dispatch_copy2d(
            self,
            dst,
            ops::Copy2dParams {
                d1,
                d2,
                src_stride: src_stride1,
                dst_stride: dst_stride1,
                src_offset,
                dst_offset,
            },
        )
        .map_err(Error::from)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> CandleResult<()> {
        wgpu_not_impl!("const_set")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_map_selects_mapped_below_threshold() {
        let device = WgpuDevice::new_test(true, 1024);
        let shape = Shape::from((128,));
        let storage = WgpuStorage::alloc(&device, &shape, DType::F32).unwrap();
        assert!(storage.is_mapped());
    }

    #[test]
    fn auto_map_selects_device_above_threshold() {
        let device = WgpuDevice::new_test(true, 64);
        let shape = Shape::from((128,));
        let storage = WgpuStorage::alloc(&device, &shape, DType::F32).unwrap();
        assert!(!storage.is_mapped());
    }

    #[test]
    fn pinned_mapped_always_mapped() {
        let device = WgpuDevice::new_test(true, 64);
        let shape = Shape::from((1024,));
        let storage = device.alloc_pinned_mapped(&shape, DType::F32).unwrap();
        assert!(storage.is_mapped());
    }
}
