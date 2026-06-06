use super::error::Result;
use super::storage::BufferOffset;
use crate::Layout;
use std::sync::{Arc, RwLock};

/// Maximum tensor rank passed to WGSL kernels via the uniform buffer.
pub const MAX_TENSOR_DIMS: usize = 8;

fn params_buffer_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
}

/// Per-tensor layout metadata mirrored in WGSL `struct TensorLayout`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorLayoutUniform {
    pub dims: [u32; MAX_TENSOR_DIMS],
    pub strides: [u32; MAX_TENSOR_DIMS],
    pub offset: u32,
    pub num_dims: u32,
    pub _pad: [u32; 2],
}

impl TensorLayoutUniform {
    pub fn from_layout(layout: &Layout) -> Self {
        let dims = layout.dims();
        let strides = layout.stride();
        let num_dims = dims.len().min(MAX_TENSOR_DIMS);

        let mut dims_arr = [0u32; MAX_TENSOR_DIMS];
        let mut strides_arr = [0u32; MAX_TENSOR_DIMS];
        for (i, &dim) in dims.iter().take(MAX_TENSOR_DIMS).enumerate() {
            dims_arr[i] = dim as u32;
        }
        for (i, &stride) in strides.iter().take(MAX_TENSOR_DIMS).enumerate() {
            strides_arr[i] = stride as u32;
        }
        Self {
            dims: dims_arr,
            strides: strides_arr,
            offset: layout.start_offset() as u32,
            num_dims: num_dims as u32,
            _pad: [0; 2],
        }
    }
}

/// Uniform block at binding 3 for standard Candle compute kernels.
///
/// WGSL shaders should declare a matching struct, e.g.:
///
/// ```wgsl
/// struct TensorLayout {
///     dims: array<u32, 8>,
///     strides: array<u32, 8>,
///     offset: u32,
///     num_dims: u32,
///     _pad: vec2<u32>,
/// }
///
/// struct Params {
///     elem_count: u32,
///     _pad: vec3<u32>,
///     out_layout: TensorLayout,
///     in0_layout: TensorLayout,
///     in1_layout: TensorLayout,
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelUniforms {
    pub elem_count: u32,
    pub _pad0: [u32; 3],
    pub out_layout: TensorLayoutUniform,
    pub in0_layout: TensorLayoutUniform,
    pub in1_layout: TensorLayoutUniform,
    pub _tail_pad: [u32; 8],
}

/// Byte size of the standard kernel params buffer (must match WGSL struct layout).
pub const STANDARD_UNIFORM_SIZE: usize = 288;

const STANDARD_UNIFORM_MIN_BINDING: std::num::NonZeroU64 = {
    match std::num::NonZeroU64::new(STANDARD_UNIFORM_SIZE as u64) {
        Some(v) => v,
        None => panic!("STANDARD_UNIFORM_SIZE must be non-zero"),
    }
};

impl KernelUniforms {
    pub fn new(
        elem_count: usize,
        out_layout: &Layout,
        in0_layout: &Layout,
        in1_layout: Option<&Layout>,
    ) -> Self {
        Self {
            elem_count: elem_count as u32,
            _pad0: [0; 3],
            out_layout: TensorLayoutUniform::from_layout(out_layout),
            in0_layout: TensorLayoutUniform::from_layout(in0_layout),
            in1_layout: in1_layout
                .map(TensorLayoutUniform::from_layout)
                .unwrap_or_default(),
            _tail_pad: [0; 8],
        }
    }

    /// Uniform block for `affine_f32`: `_pad0[0]` / `_pad0[1]` are f32 `mul` / `add` bit patterns.
    pub fn new_affine(
        elem_count: usize,
        out_layout: &Layout,
        in0_layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Self {
        let mut uniforms = Self::new(elem_count, out_layout, in0_layout, None);
        uniforms._pad0[0] = (mul as f32).to_bits();
        uniforms._pad0[1] = (add as f32).to_bits();
        uniforms
    }

    pub fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(std::mem::size_of::<Self>(), STANDARD_UNIFORM_SIZE);
        debug_assert_eq!(
            std::mem::size_of::<Self>() % wgpu::COPY_BUFFER_ALIGNMENT as usize,
            0,
            "KernelUniforms size must be a multiple of COPY_BUFFER_ALIGNMENT"
        );
        // SAFETY: `Self` is `#[repr(C)]` with no padding beyond documented fields; size is
        // checked above. Reading `size_of::<Self>()` bytes as a byte slice is valid.
        unsafe {
            std::slice::from_raw_parts((self as *const Self).cast(), std::mem::size_of::<Self>())
        }
    }
}

/// Uniform block for matrix multiply kernels (`MatMulParams` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatMulUniforms {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub c_layout: TensorLayoutUniform,
    pub a_layout: TensorLayoutUniform,
    pub b_layout: TensorLayoutUniform,
    pub _tail_pad: [u32; 8],
}

/// Uniform block for quantized matmul kernels (`QMatMulParams` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QMatMulUniforms {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _tail_pad: [u32; 68],
}

impl QMatMulUniforms {
    pub fn new(batch: usize, m: usize, n: usize, k: usize) -> Self {
        Self {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _tail_pad: [0; 68],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(std::mem::size_of::<Self>(), STANDARD_UNIFORM_SIZE);
        // SAFETY: `Self` is `#[repr(C)]` with no padding beyond documented fields; size is
        // checked above. Reading `size_of::<Self>()` bytes as a byte slice is valid.
        unsafe {
            std::slice::from_raw_parts((self as *const Self).cast(), std::mem::size_of::<Self>())
        }
    }
}

impl MatMulUniforms {
    pub fn new(
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        c_layout: &Layout,
        a_layout: &Layout,
        b_layout: &Layout,
    ) -> Self {
        Self {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            c_layout: TensorLayoutUniform::from_layout(c_layout),
            a_layout: TensorLayoutUniform::from_layout(a_layout),
            b_layout: TensorLayoutUniform::from_layout(b_layout),
            _tail_pad: [0; 8],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(std::mem::size_of::<Self>(), STANDARD_UNIFORM_SIZE);
        // SAFETY: `Self` is `#[repr(C)]` with no padding beyond documented fields; size is
        // checked above. Reading `size_of::<Self>()` bytes as a byte slice is valid.
        unsafe {
            std::slice::from_raw_parts((self as *const Self).cast(), std::mem::size_of::<Self>())
        }
    }
}

/// Uniform block for reduction kernels (`ReduceParams` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReduceUniforms {
    pub src_elem_count: u32,
    pub dst_elem_count: u32,
    pub reduce_chunk_size: u32,
    pub _pad0: u32,
    pub out_layout: TensorLayoutUniform,
    pub src_layout: TensorLayoutUniform,
    pub _unused_layout: TensorLayoutUniform,
    pub _tail_pad: [u32; 8],
}

impl ReduceUniforms {
    pub fn new(
        src_elem_count: usize,
        dst_elem_count: usize,
        reduce_chunk_size: usize,
        out_layout: &Layout,
        src_layout: &Layout,
    ) -> Self {
        Self {
            src_elem_count: src_elem_count as u32,
            dst_elem_count: dst_elem_count as u32,
            reduce_chunk_size: reduce_chunk_size as u32,
            _pad0: 0,
            out_layout: TensorLayoutUniform::from_layout(out_layout),
            src_layout: TensorLayoutUniform::from_layout(src_layout),
            _unused_layout: TensorLayoutUniform::default(),
            _tail_pad: [0; 8],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        debug_assert_eq!(std::mem::size_of::<Self>(), STANDARD_UNIFORM_SIZE);
        // SAFETY: `Self` is `#[repr(C)]` with no padding beyond documented fields; size is
        // checked above. Reading `size_of::<Self>()` bytes as a byte slice is valid.
        unsafe {
            std::slice::from_raw_parts((self as *const Self).cast(), std::mem::size_of::<Self>())
        }
    }
}

/// Cached standard bind group layout shared by Candle wgpu kernels.
#[derive(Clone)]
pub struct StandardBindGroupLayout {
    inner: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
}

impl std::fmt::Debug for StandardBindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardBindGroupLayout")
            .finish_non_exhaustive()
    }
}

impl Default for StandardBindGroupLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardBindGroupLayout {
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
            label: Some("candle standard kernel bind group layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(STANDARD_UNIFORM_MIN_BINDING),
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
}

/// Arguments for building a standard kernel bind group.
pub struct StandardBindGroupArgs<'a> {
    pub output: BufferOffset<'a>,
    pub input0: BufferOffset<'a>,
    pub input1: Option<BufferOffset<'a>>,
    pub uniforms: &'a KernelUniforms,
}

/// Builds bind groups for the standard Candle wgpu kernel layout.
#[derive(Clone)]
pub struct BindGroupBuilder {
    layout: StandardBindGroupLayout,
}

impl std::fmt::Debug for BindGroupBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindGroupBuilder").finish_non_exhaustive()
    }
}

impl Default for BindGroupBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupBuilder {
    pub fn new() -> Self {
        Self {
            layout: StandardBindGroupLayout::new(),
        }
    }

    pub fn bind_group_layout(&self) -> &StandardBindGroupLayout {
        &self.layout
    }

    pub fn create_uniform_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &KernelUniforms,
    ) -> wgpu::Buffer {
        Self::create_uniform_buffer_bytes(device, queue, uniforms.as_bytes())
    }

    pub fn create_uniform_buffer_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform_bytes: &[u8],
    ) -> wgpu::Buffer {
        debug_assert_eq!(uniform_bytes.len(), STANDARD_UNIFORM_SIZE);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle kernel uniforms"),
            size: STANDARD_UNIFORM_SIZE as u64,
            usage: params_buffer_usage(),
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, uniform_bytes);
        buffer
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        args: StandardBindGroupArgs<'_>,
    ) -> Result<wgpu::BindGroup> {
        let layout = self.layout.get_or_create(device)?;
        let uniform_buffer = self.create_uniform_buffer(device, queue, args.uniforms);

        // Unary kernels omit a second input; binding 2 still must be populated.
        let input0 = args.input0;
        let input1 = args.input1.unwrap_or(BufferOffset {
            buffer: input0.buffer,
            offset_in_bytes: input0.offset_in_bytes,
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("candle standard kernel bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: args.output.buffer,
                        offset: args.output.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input0.buffer,
                        offset: input0.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input1.buffer,
                        offset: input1.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    pub fn create_bind_group_bytes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: BufferOffset<'_>,
        input0: BufferOffset<'_>,
        input1: Option<BufferOffset<'_>>,
        uniform_bytes: &[u8],
    ) -> Result<wgpu::BindGroup> {
        let layout = self.layout.get_or_create(device)?;
        let uniform_buffer = Self::create_uniform_buffer_bytes(device, queue, uniform_bytes);
        let input1 = input1.unwrap_or(BufferOffset {
            buffer: input0.buffer,
            offset_in_bytes: input0.offset_in_bytes,
        });

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("candle kernel bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output.buffer,
                        offset: output.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input0.buffer,
                        offset: input0.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input1.buffer,
                        offset: input1.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}

/// Bind group layout for kernels with three read-only inputs (e.g. rope, where_cond).
#[derive(Clone)]
pub struct ExtendedBindGroupLayout {
    inner: Arc<RwLock<Option<wgpu::BindGroupLayout>>>,
}

impl std::fmt::Debug for ExtendedBindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtendedBindGroupLayout")
            .finish_non_exhaustive()
    }
}

impl Default for ExtendedBindGroupLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtendedBindGroupLayout {
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

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if read_only {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("candle extended kernel bind group layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(STANDARD_UNIFORM_MIN_BINDING),
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
}

/// Builds bind groups for extended (3-input) Candle wgpu kernels.
#[derive(Clone, Debug)]
pub struct ExtendedBindGroupBuilder {
    layout: ExtendedBindGroupLayout,
}

impl Default for ExtendedBindGroupBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtendedBindGroupBuilder {
    pub fn new() -> Self {
        Self {
            layout: ExtendedBindGroupLayout::new(),
        }
    }

    pub fn bind_group_layout(&self) -> &ExtendedBindGroupLayout {
        &self.layout
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: BufferOffset<'_>,
        input0: BufferOffset<'_>,
        input1: BufferOffset<'_>,
        input2: BufferOffset<'_>,
        uniform_bytes: &[u8],
    ) -> Result<wgpu::BindGroup> {
        let layout = self.layout.get_or_create(device)?;
        let uniform_buffer = BindGroupBuilder::create_uniform_buffer_bytes(device, queue, uniform_bytes);

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("candle extended kernel bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output.buffer,
                        offset: output.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input0.buffer,
                        offset: input0.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input1.buffer,
                        offset: input1.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input2.buffer,
                        offset: input2.offset_in_bytes,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}

macro_rules! fixed_uniform {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? } pad $pad:expr) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $(pub $field: $ty,)*
            pub _pad: [u32; $pad],
        }

        impl $name {
            pub fn as_bytes(&self) -> &[u8] {
                debug_assert_eq!(std::mem::size_of::<Self>(), STANDARD_UNIFORM_SIZE);
                // SAFETY: `Self` is `#[repr(C)]` with fixed size `STANDARD_UNIFORM_SIZE`.
                unsafe {
                    std::slice::from_raw_parts(
                        (self as *const Self).cast(),
                        std::mem::size_of::<Self>(),
                    )
                }
            }
        }
    };
}

fixed_uniform!(Copy2dUniforms {
    d1: u32,
    d2: u32,
    src_stride: u32,
    dst_stride: u32,
    src_offset: u32,
    dst_offset: u32,
} pad 66);

fixed_uniform!(RmsNormUniforms {
    n_rows: u32,
    n_cols: u32,
    eps_bits: u32,
} pad 69);

fixed_uniform!(RopeUniforms {
    b: u32,
    h: u32,
    t: u32,
    d: u32,
    unbatched_cs: u32,
} pad 67);

fixed_uniform!(WhereUniforms { elem_count: u32 } pad 71);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_uniforms_pack_mul_add_into_pad0() {
        let layout = Layout::contiguous((2, 3));
        let u = KernelUniforms::new_affine(6, &layout, &layout, 2.0, 0.5);
        assert_eq!(u._pad0[0], 2.0f32.to_bits());
        assert_eq!(u._pad0[1], 0.5f32.to_bits());
    }

    #[test]
    fn kernel_uniforms_size_is_copy_aligned() {
        assert_eq!(std::mem::size_of::<TensorLayoutUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<KernelUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<KernelUniforms>(), STANDARD_UNIFORM_SIZE);
        assert_eq!(std::mem::size_of::<MatMulUniforms>(), STANDARD_UNIFORM_SIZE);
        assert_eq!(std::mem::size_of::<QMatMulUniforms>(), STANDARD_UNIFORM_SIZE);
        assert_eq!(std::mem::size_of::<ReduceUniforms>(), STANDARD_UNIFORM_SIZE);
    }

    #[test]
    fn tensor_layout_from_candle_layout() {
        let layout = Layout::contiguous((2, 3));
        let uniform = TensorLayoutUniform::from_layout(&layout);
        assert_eq!(uniform.num_dims, 2);
        assert_eq!(uniform.dims[..2], [2, 3]);
        assert_eq!(uniform.strides[..2], [3, 1]);
    }
}
