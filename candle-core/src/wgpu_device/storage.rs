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

/// Binding view into a wgpu storage buffer.
///
/// `offset_in_bytes` is always zero: WGSL kernels index from the buffer start using
/// element offsets in the uniform block (`TensorLayoutUniform::offset`, etc.).
/// Non-zero bind offsets would violate `min_storage_buffer_offset_alignment`.
#[derive(Debug, Clone)]
pub struct BufferOffset<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub offset_in_bytes: u64,
}

/// Returns a storage-buffer binding for `storage`/`layout`.
///
/// The logical element offset lives in kernel uniforms, not in the bind-group offset.
pub fn buffer_offset<'a>(storage: &'a WgpuStorage, _layout: &Layout) -> BufferOffset<'a> {
    BufferOffset {
        buffer: storage.backing.buffer(),
        offset_in_bytes: 0,
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

    /// Uploads a raw byte slice into a freshly allocated GPU buffer (single host→GPU copy).
    pub fn from_bytes(
        device: &WgpuDevice,
        data: &[u8],
        elem_count: usize,
        dtype: DType,
    ) -> Result<Self> {
        let expected = elem_count * dtype.size_in_bytes();
        if data.len() != expected {
            return Err(WgpuError::Message(format!(
                "wgpu from_bytes: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        let byte_len = data.len();
        let use_mapped = Self::should_use_mapped(device, byte_len, false);
        let backing = if use_mapped {
            let mapped = MappedBacking::new(device, byte_len)?;
            device.queue().write_buffer(mapped.storage(), 0, data);
            BufferBacking::Mapped(mapped)
        } else {
            let buffer = device.allocate_buffer(byte_len, STORAGE_BUFFER_USAGE)?;
            device.queue().write_buffer(&buffer, 0, data);
            BufferBacking::DeviceLocal(buffer)
        };
        Ok(Self::new(backing, device.clone(), elem_count, dtype))
    }

    /// Uploads CPU storage into a freshly allocated GPU buffer.
    pub fn from_cpu(device: &WgpuDevice, storage: &CpuStorage) -> Result<Self> {
        let dtype = storage.dtype();
        let (bytes, elem_count) = cpu_storage_as_bytes(storage)?;
        Self::from_bytes(device, bytes, elem_count, dtype)
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

    fn cast_via_cpu(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu = self
            .to_cpu_storage()
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        let casted = cpu
            .to_dtype(layout, dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
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
        let elem_count = out_layout.shape().elem_count();
        let (wg_size, grid) = cast_dispatch_params(entry, elem_count);
        let kernel = WgpuKernel::compile_with_workgroup_size(&self.device, CAST, entry, wg_size)?;
        let uniforms =
            super::bind_group::KernelUniforms::new(elem_count, &out_layout, layout, None);
        let bind_group_builder = super::bind_group::BindGroupBuilder::new();
        let bind_group = bind_group_builder.create_bind_group_bytes(
            self.device.device(),
            self.device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(self, layout),
            None,
            uniforms.as_bytes(),
        )?;
        self.backing.with_unmapped(|| {
            kernel.dispatch_bind_group(&self.device, &bind_group, [grid, 1, 1])
        })?;
        Ok(out)
    }
}

/// GPU buffer-to-buffer copy for contiguous regions (requires 4-byte alignment).
pub(crate) fn copy_buffer_region(
    device: &WgpuDevice,
    src: &BufferBacking,
    dst: &BufferBacking,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) -> Result<()> {
    const ALIGN: u64 = wgpu::COPY_BUFFER_ALIGNMENT;
    if src_offset % ALIGN != 0 || dst_offset % ALIGN != 0 || size % ALIGN != 0 {
        return Err(WgpuError::Message(format!(
            "copy_buffer_region requires {ALIGN}-byte alignment (src={src_offset}, dst={dst_offset}, size={size})"
        )));
    }
    src.with_unmapped(|| {
        dst.with_unmapped(|| {
            let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu copy_buffer_region"),
            });
            encoder.copy_buffer_to_buffer(
                src.buffer(),
                src_offset,
                dst.buffer(),
                dst_offset,
                size,
            );
            device.queue().submit(Some(encoder.finish()));
            Ok(())
        })?;
        Ok(())
    })
}

fn read_bytes_staging(device: &WgpuDevice, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    super::async_io::poll_device(device.device())?;
    let wgpu_device = device.device();
    let queue = device.queue();

    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let copy_size = size.div_ceil(align) * align;

    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu readback staging"),
        size: copy_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu readback"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, copy_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    wait_for_buffer_map(wgpu_device, &rx)?;

    let mapped = slice.get_mapped_range();
    let mut bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    bytes.truncate(size as usize);
    Ok(bytes)
}

const CAST_WG_SIZE: u32 = 32;

/// Workgroup size and X grid dimension for parallel cast kernels.
fn cast_dispatch_params(entry: &str, elem_count: usize) -> (u32, u32) {
    let parallel_elems = match entry {
        "cast_f16_f32" | "cast_f32_f16" | "cast_bf16_f32" | "cast_f32_bf16" | "cast_f16_bf16"
        | "cast_bf16_f16" => elem_count.div_ceil(2),
        "cast_u8_f32" | "cast_f32_u8" => elem_count.div_ceil(4),
        _ => elem_count,
    };
    (
        CAST_WG_SIZE,
        super::kernel::workgroup_count(CAST_WG_SIZE, parallel_elems),
    )
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
        (DType::F32, DType::U32) => Some("cast_f32_u32"),
        _ => None,
    }
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

pub(crate) fn typed_slice_as_bytes<T>(data: &[T]) -> &[u8] {
    bytemuck_slice(data)
}

fn bytemuck_slice<T>(data: &[T]) -> &[u8] {
    // SAFETY: Reinterpreting a contiguous slice of plain-old-data elements as bytes is
    // valid for the numeric tensor dtypes uploaded to wgpu.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
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

    fn powf(&self, layout: &Layout, exp: f64) -> CandleResult<Self> {
        ops::dispatch_powf(self, layout, exp)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> CandleResult<Self> {
        ops::dispatch_elu(self, layout, alpha)
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> CandleResult<Self> {
        ops::dispatch_reduce(self, op, layout, reduce_dims)
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> CandleResult<Self> {
        ops::dispatch_cmp(self, rhs, lhs_layout, rhs_layout, op)
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
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> CandleResult<Self> {
        super::conv::conv1d(self, layout, kernel, kernel_l, params)
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> CandleResult<Self> {
        super::conv::conv_transpose1d(self, layout, kernel, kernel_l, params)
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> CandleResult<Self> {
        super::conv::conv2d(self, layout, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> CandleResult<Self> {
        super::conv::conv_transpose2d(self, layout, kernel, kernel_l, params)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CandleResult<Self> {
        super::conv::avg_pool2d(self, layout, kernel_size, stride)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CandleResult<Self> {
        super::conv::max_pool2d(self, layout, kernel_size, stride)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> CandleResult<Self> {
        super::conv::upsample_nearest1d(self, layout, sz)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> CandleResult<Self> {
        super::conv::upsample_nearest2d(self, layout, h, w)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> CandleResult<Self> {
        super::conv::upsample_bilinear2d(self, layout, h, w, align_corners, scale_h, scale_w)
    }

    fn gather(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
    ) -> CandleResult<Self> {
        super::indexing::dispatch_gather_f32(self, layout, ids, ids_l, dim)
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> CandleResult<()> {
        super::indexing::dispatch_scatter_f32(self, layout, ids, ids_l, src, src_l, dim, false)
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> CandleResult<()> {
        super::indexing::dispatch_scatter_f32(self, layout, ids, ids_l, src, src_l, dim, true)
    }

    fn index_select(
        &self,
        ids: &Self,
        layout: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> CandleResult<Self> {
        super::indexing::dispatch_index_select_f32(self, layout, ids, ids_l, dim)
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> CandleResult<Self> {
        super::indexing::dispatch_index_add_f32(self, layout, ids, ids_l, src, src_l, dim)
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

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> CandleResult<()> {
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

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> CandleResult<()> {
        super::fill::dispatch_const_set(self, layout, scalar)
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

    #[test]
    fn from_bytes_round_trip() {
        let device = WgpuDevice::new_test(true, 1024);
        let data = [1.0f32, 2.0, 3.0];
        let bytes = typed_slice_as_bytes(&data);
        let storage = WgpuStorage::from_bytes(&device, bytes, 3, DType::F32).unwrap();
        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.elem_count(), 3);
    }

    #[test]
    fn buffer_offset_binds_at_zero_with_nonzero_layout() {
        let device = WgpuDevice::new_test(true, 1024);
        let storage = WgpuStorage::alloc(&device, &Shape::from((20,)), DType::F32).unwrap();
        // start_offset=9 → byte offset 36 would fail min_storage_buffer_offset_alignment=32
        let layout = Layout::contiguous_with_offset((3,), 9);
        let binding = buffer_offset(&storage, &layout);
        assert_eq!(binding.offset_in_bytes, 0);
        assert!(std::ptr::eq(
            binding.buffer,
            storage.backing.buffer().as_ref()
        ));
    }

    #[test]
    fn buffer_offset_zero_for_unaligned_dtypes() {
        let device = WgpuDevice::new_test(true, 1024);
        let cases = [
            (Shape::from((20,)), DType::F32, 9usize),
            (Shape::from((20,)), DType::F16, 9usize),
            (Shape::from((20,)), DType::BF16, 9usize),
            (Shape::from((40,)), DType::U32, 9usize),
            (Shape::from((40,)), DType::U8, 36usize),
        ];
        for (shape, dtype, start_offset) in cases {
            let storage = WgpuStorage::alloc(&device, &shape, dtype).unwrap();
            let layout = Layout::contiguous_with_offset((3,), start_offset);
            let binding = buffer_offset(&storage, &layout);
            assert_eq!(
                binding.offset_in_bytes, 0,
                "dtype {dtype:?} start_offset {start_offset}"
            );
        }
    }

    #[test]
    fn device_reports_storage_buffer_offset_alignment() {
        let device = WgpuDevice::new_test(true, 1024);
        assert!(device.storage_buffer_offset_alignment() > 0);
        assert_eq!(
            device.storage_buffer_offset_alignment(),
            device.device().limits().min_storage_buffer_offset_alignment
        );
    }

    #[test]
    fn cast_f64_to_f32_noop_device() {
        let device = WgpuDevice::new_test(true, 1024);
        let storage = WgpuStorage::from_cpu(
            &device,
            &CpuStorage::F64(vec![1.0, 2.5, std::f64::consts::PI]),
        )
        .unwrap();
        let layout = Layout::contiguous(&Shape::from((3,)));
        let f32 = storage.to_dtype(&layout, DType::F32).unwrap();
        let cpu = f32.to_cpu_storage().unwrap();
        let CpuStorage::F32(v) = cpu else {
            panic!("expected f32");
        };
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.5).abs() < 1e-6);
        assert!((v[2] - std::f32::consts::PI).abs() < 1e-5);
    }
}
