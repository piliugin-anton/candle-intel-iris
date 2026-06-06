use super::bind_group::{
    BindGroupBuilder, Copy2dUniforms, KernelUniforms, MatMulUniforms,
    QMatMulUniforms, ReduceUniforms, RmsNormUniforms, RopeUniforms, WhereUniforms,
};
use super::error::{Result, WgpuError};
use super::intel_caps::{tune_matmul_shader_source, IntelGeneration};
use super::kernel::WgpuKernel;
use super::storage::{buffer_offset, BufferOffset, WgpuStorage};
use super::WgpuDevice;
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, ReduceOp, UnaryOpT};
use crate::wgsl::{
    BINARY, COPY, COPY2D, MATMUL_NAIVE, MATMUL_TILED, MATMUL_TILED_F16, MATMUL_TILED_VEC,
    MATMUL_VEC_WIDTH, QMATMUL_Q4_0, REDUCE, RMS_NORM, ROPE, UNARY, WHERE_COND,
};
use crate::{CpuStorage, DType, Error, Layout, Result as CandleResult, Shape};
use std::sync::Arc;
use wgpu::BufferUsages;

fn layout_has_broadcast(layout: &Layout) -> bool {
    layout.stride().contains(&0)
}

/// Expands broadcast/strided layouts to contiguous GPU storage for elem-wise kernels.
fn materialize_if_needed(
    storage: &WgpuStorage,
    layout: &Layout,
    device: &WgpuDevice,
) -> Result<(WgpuStorage, Layout)> {
    if !layout_has_broadcast(layout) && layout.is_contiguous() {
        return Ok((storage.clone(), layout.clone()));
    }
    let cpu = storage
        .to_cpu_storage()
        .map_err(|e| WgpuError::Message(e.to_string()))?;
    let CpuStorage::F32(src) = cpu else {
        return Err(WgpuError::Message(
            "wgpu materialize only supports f32".into(),
        ));
    };
    let shape = layout.shape();
    let dims = shape.dims();
    let strides = layout.stride();
    let start = layout.start_offset();
    let mut dst = vec![0f32; shape.elem_count()];

    // Fast path: one broadcast dimension with stride 0 (common for softmax).
    if dims.len() == 2 && strides[1] == 0 {
        let n = dims[0];
        let m = dims[1];
        let row_stride = strides[0];
        for i in 0..n {
            let v = src[start + i * row_stride];
            let base = i * m;
            for j in 0..m {
                dst[base + j] = v;
            }
        }
    } else if dims.len() == 3 && strides[2] == 0 {
        let d1 = dims[1];
        let d2 = dims[2];
        let row_stride = strides[1];
        for j in 0..d1 {
            let v = src[start + j * row_stride];
            let base = j * d2;
            for k in 0..d2 {
                dst[base + k] = v;
            }
        }
    } else {
        for (flat_i, src_i) in layout.strided_index().enumerate() {
            dst[flat_i] = src[src_i];
        }
    }
    let out = WgpuStorage::from_cpu(device, &CpuStorage::F32(dst))?;
    device
        .synchronize()
        .map_err(|e| WgpuError::Message(e.to_string()))?;
    Ok((out, Layout::contiguous(shape)))
}

fn require_f32(dtype: DType, op: &'static str) -> CandleResult<()> {
    if dtype == DType::F32 {
        Ok(())
    } else {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    }
}

fn require_matmul_dtype(dtype: DType, op: &'static str) -> CandleResult<()> {
    match dtype {
        DType::F32 | DType::F16 => Ok(()),
        other => Err(Error::UnsupportedDTypeForOp(other, op).bt()),
    }
}

/// Selected matrix-multiply kernel variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatMulKernel {
    NaiveF32,
    TiledF32,
    TiledVecF32,
    TiledF16,
    TiledVecF16,
}

fn select_matmul_kernel(
    device: &WgpuDevice,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
) -> MatMulKernel {
    let caps = device.caps();
    let compute_dtype = caps.effective_compute_dtype(dtype);
    let tiled = matches!(
        caps.generation,
        IntelGeneration::Gen12Plus | IntelGeneration::Gen11
    );
    let vec_aligned = k.is_multiple_of(MATMUL_VEC_WIDTH as usize)
        && n.is_multiple_of(MATMUL_VEC_WIDTH as usize);
    let large_enough = m >= 16 || n >= 16 || k >= 16;

    match compute_dtype {
        DType::F16 if caps.supports_native_f16() => {
            if tiled && vec_aligned && large_enough {
                MatMulKernel::TiledVecF16
            } else {
                MatMulKernel::TiledF16
            }
        }
        _ => {
            if tiled && vec_aligned && large_enough {
                MatMulKernel::TiledVecF32
            } else if tiled {
                MatMulKernel::TiledF32
            } else {
                MatMulKernel::NaiveF32
            }
        }
    }
}

fn matmul_shader_source(kernel: MatMulKernel) -> &'static str {
    match kernel {
        MatMulKernel::NaiveF32 => MATMUL_NAIVE,
        MatMulKernel::TiledF32 => MATMUL_TILED,
        MatMulKernel::TiledVecF32 => MATMUL_TILED_VEC,
        MatMulKernel::TiledF16 | MatMulKernel::TiledVecF16 => MATMUL_TILED_F16,
    }
}

fn matmul_entry_point(kernel: MatMulKernel) -> &'static str {
    match kernel {
        MatMulKernel::NaiveF32 => "matmul_naive_f32",
        MatMulKernel::TiledF32 => "matmul_tiled_f32",
        MatMulKernel::TiledVecF32 => "matmul_tiled_vec_f32",
        MatMulKernel::TiledF16 => "matmul_tiled_f16",
        MatMulKernel::TiledVecF16 => "matmul_tiled_vec_f16",
    }
}

fn unary_entry_point(kernel: &str) -> Option<&'static str> {
    match kernel {
        "uexp" => Some("exp_f32"),
        "uneg" => Some("neg_f32"),
        "ulog" => Some("log_f32"),
        "usqrt" => Some("sqrt_f32"),
        "urelu" => Some("relu_f32"),
        "urecip" => Some("recip_f32"),
        "usilu" => Some("silu_f32"),
        "ugelu" => Some("gelu_f32"),
        "uabs" => Some("abs_f32"),
        _ => None,
    }
}

fn binary_entry_point(kernel: &str) -> Option<&'static str> {
    match kernel {
        "badd" => Some("add_f32"),
        "bsub" => Some("sub_f32"),
        "bmul" => Some("mul_f32"),
        "bdiv" => Some("div_f32"),
        "bminimum" => Some("min_f32"),
        "bmaximum" => Some("max_f32"),
        _ => None,
    }
}

fn reduce_entry_point(op: ReduceOp) -> Option<&'static str> {
    match op {
        ReduceOp::Sum => Some("reduce_sum_f32"),
        ReduceOp::Max => Some("reduce_max_f32"),
        ReduceOp::Min => Some("reduce_min_f32"),
        ReduceOp::ArgMax => Some("reduce_argmax_f32"),
        ReduceOp::ArgMin => None,
    }
}

fn compile_elemwise_kernel(
    device: &WgpuDevice,
    source: &str,
    entry_point: &str,
) -> Result<WgpuKernel> {
    // Elem-wise shaders use @workgroup_size(1) with one workgroup per element.
    WgpuKernel::compile_with_workgroup_size(device, source, entry_point, 1)
}

struct ElemwiseDispatch<'a> {
    device: &'a WgpuDevice,
    source: &'a str,
    entry_point: &'a str,
    out: &'a WgpuStorage,
    out_layout: &'a Layout,
    in0: &'a WgpuStorage,
    in0_layout: &'a Layout,
    in1: Option<(&'a WgpuStorage, &'a Layout)>,
    backing: &'a super::storage::BufferBacking,
}

fn dispatch_elemwise(args: ElemwiseDispatch<'_>) -> Result<()> {
    let ElemwiseDispatch {
        device,
        source,
        entry_point,
        out,
        out_layout,
        in0,
        in0_layout,
        in1,
        backing,
    } = args;
    let kernel = compile_elemwise_kernel(device, source, entry_point)?;
    let uniforms = KernelUniforms::new(
        out_layout.shape().elem_count(),
        out_layout,
        in0_layout,
        in1.map(|(_, l)| l),
    );
    let bind_group_builder = BindGroupBuilder::new();
    let in0_offset = buffer_offset(in0, in0_layout);
    let in1_offset = in1.map(|(storage, layout)| buffer_offset(storage, layout));
    let bind_group = bind_group_builder.create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(out, out_layout),
        in0_offset.clone(),
        in1_offset,
        uniforms.as_bytes(),
    )?;
    let elem_count = out_layout.shape().elem_count() as u32;
    backing.with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [elem_count, 1, 1]))
}

fn compile_matmul_kernel(device: &WgpuDevice, kernel: MatMulKernel) -> Result<WgpuKernel> {
    let source = matmul_shader_source(kernel);
    let entry = matmul_entry_point(kernel);
    let tuned = tune_matmul_shader_source(source, device.caps());
    let tile = device.caps().matmul_tile_size;
    WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, tile)
}

fn compile_reduce_kernel(device: &WgpuDevice, entry_point: &str) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(REDUCE, device.caps());
    WgpuKernel::compile_with_workgroup_size(
        device,
        &tuned,
        entry_point,
        device.caps().reduce_workgroup_size,
    )
}

pub fn dispatch_affine(
    storage: &WgpuStorage,
    layout: &Layout,
    mul: f64,
    add: f64,
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), "affine")?;

    let device = storage.device().clone();
    let out_shape = layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::F32)?;
    let out_layout = Layout::contiguous(out_shape);
    let uniforms = KernelUniforms::new_affine(
        out_layout.shape().elem_count(),
        &out_layout,
        layout,
        mul,
        add,
    );
    let kernel = compile_elemwise_kernel(&device, UNARY, "affine_f32")?;
    let bind_group_builder = BindGroupBuilder::new();
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(storage, layout),
            None,
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    let elem_count = out_layout.shape().elem_count() as u32;
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(&device, &bind_group, [elem_count, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_unary<B: UnaryOpT>(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), B::NAME)?;
    let entry = unary_entry_point(B::KERNEL).ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu unary kernel not available for op {}",
            B::NAME
        ))
    })?;

    let device = storage.device().clone();
    let out_shape = layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::F32)?;
    let out_layout = Layout::contiguous(out_shape);
    dispatch_elemwise(ElemwiseDispatch {
        device: &device,
        source: UNARY,
        entry_point: entry,
        out: &out,
        out_layout: &out_layout,
        in0: storage,
        in0_layout: layout,
        in1: None,
        backing: storage.backing(),
    })
    .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_binary<B: BinaryOpT>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(lhs.dtype(), B::NAME)?;
    require_f32(rhs.dtype(), B::NAME)?;
    let entry = binary_entry_point(B::KERNEL).ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu binary kernel not available for op {}",
            B::NAME
        ))
    })?;

    let device = lhs.device().clone();
    let (lhs, lhs_layout) = materialize_if_needed(lhs, lhs_layout, &device).map_err(Error::from)?;
    let (rhs, rhs_layout) = materialize_if_needed(rhs, rhs_layout, &device).map_err(Error::from)?;
    let out_shape = lhs_layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::F32)?;
    let out_layout = Layout::contiguous(out_shape);
    dispatch_elemwise(ElemwiseDispatch {
        device: &device,
        source: BINARY,
        entry_point: entry,
        out: &out,
        out_layout: &out_layout,
        in0: &lhs,
        in0_layout: &lhs_layout,
        in1: Some((&rhs, &rhs_layout)),
        backing: lhs.backing(),
    })
    .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_matmul(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_matmul_dtype(lhs.dtype(), "matmul")?;
    require_matmul_dtype(rhs.dtype(), "matmul")?;
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::UnsupportedDTypeForOp(lhs.dtype(), "matmul").bt());
    }

    let device = lhs.device().clone();
    let storage_dtype = lhs.dtype();
    let compute_dtype = device.caps().effective_compute_dtype(storage_dtype);

    if compute_dtype == DType::F32 && storage_dtype == DType::F16 {
        let lhs_f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
        let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
        let lhs_layout = Layout::contiguous(lhs_layout.shape());
        let rhs_layout = Layout::contiguous(rhs_layout.shape());
        let out_shape = {
            let lhs_dims = lhs_layout.shape().dims();
            let dim = lhs_dims.len();
            let mut out_dims = lhs_dims[..dim - 2].to_vec();
            out_dims.push(m);
            out_dims.push(n);
            Shape::from(out_dims)
        };
        let out_f32 = dispatch_matmul_inner(
            &lhs_f32,
            &rhs_f32,
            (b, m, n, k),
            &lhs_layout,
            &rhs_layout,
            DType::F32,
        )?;
        let out_layout = Layout::contiguous(&out_shape);
        return out_f32.to_dtype(&out_layout, DType::F16);
    }

    dispatch_matmul_inner(lhs, rhs, (b, m, n, k), lhs_layout, rhs_layout, storage_dtype)
}

fn dispatch_matmul_inner(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    out_dtype: DType,
) -> CandleResult<WgpuStorage> {
    let device = lhs.device().clone();
    let lhs_dims = lhs_layout.shape().dims();
    let dim = lhs_dims.len();
    let mut out_dims = lhs_dims[..dim - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);
    let out_layout = Layout::contiguous(&out_shape);
    let out = WgpuStorage::alloc(&device, &out_shape, out_dtype)?;

    let uniforms = MatMulUniforms::new(b, m, n, k, &out_layout, lhs_layout, rhs_layout);
    let kernel_kind = select_matmul_kernel(&device, out_dtype, m, n, k);
    let kernel = compile_matmul_kernel(&device, kernel_kind).map_err(Error::from)?;
    let bind_group_builder = BindGroupBuilder::new();
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(lhs, lhs_layout),
            Some(buffer_offset(rhs, rhs_layout)),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;

    let wg = device.caps().matmul_tile_size;
    let grid_x = (n as u32).div_ceil(wg);
    let grid_y = (m as u32).div_ceil(wg);
    let grid_z = b as u32;

    lhs.backing()
        .with_unmapped(|| {
            kernel.dispatch_bind_group(&device, &bind_group, [grid_x, grid_y, grid_z])
        })
        .map_err(Error::from)?;
    Ok(out)
}

fn compile_qmatmul_kernel(device: &WgpuDevice) -> Result<WgpuKernel> {
    WgpuKernel::compile_with_workgroup_size(device, QMATMUL_Q4_0, "qmatmul_q4_0_f32", 8)
}

/// Quantized Q4_0 matrix multiply: `dst = lhs @ rhs^T` with f32 activations.
///
/// `rhs` is a raw Q4_0 buffer (GGML block layout, shape n × k).
pub fn dispatch_qmatmul_q4_0(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(lhs.dtype(), "qmatmul")?;
    if !k.is_multiple_of(32) {
        return Err(WgpuError::Message(format!(
            "qmatmul k={k} must be divisible by 32"
        ))
        .into());
    }

    let device = lhs.device().clone();
    let lhs_dims = lhs_layout.shape().dims();
    let dim = lhs_dims.len();
    let mut out_dims = lhs_dims[..dim - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);
    let out_layout = Layout::contiguous(&out_shape);
    let out = WgpuStorage::alloc(&device, &out_shape, DType::F32)?;

    let uniforms = QMatMulUniforms::new(b, m, n, k);
    let kernel = compile_qmatmul_kernel(&device).map_err(Error::from)?;
    let bind_group_builder = BindGroupBuilder::new();
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(lhs, lhs_layout),
            Some(BufferOffset {
                buffer: rhs_buffer,
                offset_in_bytes: 0,
            }),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;

    let grid_x = n as u32;
    let grid_y = m as u32;
    let grid_z = b as u32;

    lhs.backing()
        .with_unmapped(|| {
            kernel.dispatch_bind_group(&device, &bind_group, [grid_x, grid_y, grid_z])
        })
        .map_err(Error::from)?;
    Ok(out)
}

/// Upload Q4_0 weight bytes to a GPU storage buffer.
pub fn upload_q4_0_weights(device: &WgpuDevice, bytes: &[u8]) -> Result<Arc<wgpu::Buffer>> {
    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("candle q4_0 weights"),
        size: bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device.queue().write_buffer(&buffer, 0, bytes);
    Ok(Arc::new(buffer))
}

pub fn dispatch_reduce(
    storage: &WgpuStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), "reduce")?;

    if matches!(op, ReduceOp::ArgMin) {
        return Err(
            WgpuError::Message(format!("wgpu reduce op {:?} not yet implemented", op)).into(),
        );
    }

    if matches!(op, ReduceOp::ArgMax) && reduce_dims.len() != 1 {
        return Err(Error::OnlySingleDimension {
            op: "argmax",
            dims: reduce_dims.to_vec(),
        }
        .bt());
    }

    let entry = reduce_entry_point(op)
        .ok_or_else(|| WgpuError::Message(format!("wgpu reduce op {:?} not available", op)))?;

    let src_dims = layout.dims();
    let mut dst_dims = src_dims.to_vec();
    for &dim in reduce_dims {
        dst_dims[dim] = 1;
    }
    let dst_shape = Shape::from(dst_dims);
    let dst_layout = Layout::contiguous(&dst_shape);

    let src_elem_count = layout.shape().elem_count();
    let dst_elem_count = dst_shape.elem_count();
    if dst_elem_count == 0 {
        return WgpuStorage::alloc(storage.device(), &dst_shape, DType::F32).map_err(Error::from);
    }
    let reduce_chunk_size = src_elem_count / dst_elem_count;

    let device = storage.device().clone();
    let out = WgpuStorage::alloc(&device, &dst_shape, DType::F32)?;
    let uniforms = ReduceUniforms::new(
        src_elem_count,
        dst_elem_count,
        reduce_chunk_size,
        &dst_layout,
        layout,
    );

    let kernel = compile_reduce_kernel(&device, entry)?;
    let bind_group_builder = BindGroupBuilder::new();
    let in0 = buffer_offset(storage, layout);
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &dst_layout),
            in0.clone(),
            Some(in0),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;

    storage
        .backing()
        .with_unmapped(|| {
            kernel.dispatch_bind_group(&device, &bind_group, [dst_elem_count as u32, 1, 1])
        })
        .map_err(Error::from)?;
    Ok(out)
}

fn compile_copy_kernel(device: &WgpuDevice, source: &str, entry: &str) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(source, device.caps());
    WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, device.caps().elem_workgroup_size)
}

pub fn dispatch_copy_strided_src(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    dst_offset: usize,
    src_layout: &Layout,
) -> Result<()> {
    if src.dtype() != DType::F32 || dst.dtype() != DType::F32 {
        return Err(WgpuError::Message("wgpu copy_strided_src only supports f32".into()));
    }
    let device = src.device();
    let elem_count = src_layout.shape().elem_count();
    let dst_shape = Shape::from(elem_count);
    let dst_layout = Layout::new(dst_shape.clone(), vec![1], dst_offset);
    let uniforms = KernelUniforms::new(elem_count, &dst_layout, src_layout, None);
    let kernel = compile_copy_kernel(device, COPY, "copy_strided_f32")?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(dst, &dst_layout),
        buffer_offset(src, src_layout),
        None,
        uniforms.as_bytes(),
    )?;
    let wg = device.caps().elem_workgroup_size;
    let grid = (elem_count as u32).div_ceil(wg);
    dst.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

pub fn dispatch_copy2d(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    d1: usize,
    d2: usize,
    src_stride: usize,
    dst_stride: usize,
    src_offset: usize,
    dst_offset: usize,
) -> Result<()> {
    if src.dtype() != DType::F32 || dst.dtype() != DType::F32 {
        return Err(WgpuError::Message("wgpu copy2d only supports f32".into()));
    }
    let device = src.device();
    let uniforms = Copy2dUniforms {
        d1: d1 as u32,
        d2: d2 as u32,
        src_stride: src_stride as u32,
        dst_stride: dst_stride as u32,
        src_offset: src_offset as u32,
        dst_offset: dst_offset as u32,
        _pad: [0; 66],
    };
    let kernel = WgpuKernel::compile_with_workgroup_size(
        device,
        COPY2D,
        "copy2d_f32",
        device.caps().elem_workgroup_size,
    )?;
    let dummy = Layout::contiguous((1,));
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(dst, &dummy),
        buffer_offset(src, &dummy),
        None,
        uniforms.as_bytes(),
    )?;
    let total = (d1 * d2) as u32;
    let wg = device.caps().elem_workgroup_size;
    let grid = total.div_ceil(wg);
    dst.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

pub fn dispatch_rms_norm_f32(
    x: &WgpuStorage,
    alpha: &WgpuStorage,
    x_layout: &Layout,
    alpha_layout: &Layout,
    eps: f32,
) -> CandleResult<WgpuStorage> {
    require_f32(x.dtype(), "rms_norm")?;
    require_f32(alpha.dtype(), "rms_norm")?;
    let device = x.device();
    let dims = x_layout.dims();
    let n_cols = *dims
        .last()
        .ok_or_else(|| Error::Msg("empty tensor in rms_norm".into()))?;
    let n_rows = x_layout.shape().elem_count() / n_cols;
    let out = WgpuStorage::alloc(device, x_layout.shape(), DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(x_layout.shape());
    let uniforms = RmsNormUniforms {
        n_rows: n_rows as u32,
        n_cols: n_cols as u32,
        eps_bits: eps.to_bits(),
        _pad: [0; 69],
    };
    let kernel = WgpuKernel::compile_with_workgroup_size(device, RMS_NORM, "rms_norm_f32", 32)
        .map_err(Error::from)?;
    let bind_group = BindGroupBuilder::new()
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            buffer_offset(x, x_layout),
            Some(buffer_offset(alpha, alpha_layout)),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    x.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [n_rows as u32, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_rope_f32(
    src: &WgpuStorage,
    cos: &WgpuStorage,
    sin: &WgpuStorage,
    src_layout: &Layout,
    cos_layout: &Layout,
    sin_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(src.dtype(), "rope")?;
    require_f32(cos.dtype(), "rope")?;
    require_f32(sin.dtype(), "rope")?;
    let (b, h, t, d) = src_layout.shape().dims4()?;
    let out = WgpuStorage::alloc(src.device(), src_layout.shape(), DType::F32)
        .map_err(Error::from)?;
    let out_layout = Layout::contiguous(src_layout.shape());
    let unbatched_cs = cos_layout.dims().len() == 3 && sin_layout.dims().len() == 3;
    let uniforms = RopeUniforms {
        b: b as u32,
        h: h as u32,
        t: t as u32,
        d: d as u32,
        unbatched_cs: u32::from(unbatched_cs),
        _pad: [0; 67],
    };
    let device = src.device();
    let kernel =
        WgpuKernel::compile_extended(device, ROPE, "rope_f32", 32).map_err(Error::from)?;
    let bind_group = kernel
        .create_extended_bind_group(
            device,
            buffer_offset(&out, &out_layout),
            buffer_offset(src, src_layout),
            buffer_offset(cos, cos_layout),
            buffer_offset(sin, sin_layout),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    let grid = (b * h * t) as u32;
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_where_u8_f32(
    cond: &WgpuStorage,
    on_true: &WgpuStorage,
    on_false: &WgpuStorage,
    cond_layout: &Layout,
    on_true_layout: &Layout,
    on_false_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    if cond.dtype() != DType::U8 {
        return Err(
            Error::UnsupportedDTypeForOp(cond.dtype(), "where_cond predicate").bt(),
        );
    }
    require_f32(on_true.dtype(), "where_cond")?;
    require_f32(on_false.dtype(), "where_cond")?;
    let device = cond.device();
    let elem_count = cond_layout.shape().elem_count();
    let out = WgpuStorage::alloc(device, cond_layout.shape(), DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(cond_layout.shape());
    let uniforms = WhereUniforms {
        elem_count: elem_count as u32,
        _pad: [0; 71],
    };
    let kernel = WgpuKernel::compile_extended(device, WHERE_COND, "where_u8_f32", 32)
        .map_err(Error::from)?;
    let bind_group = kernel
        .create_extended_bind_group(
            device,
            buffer_offset(&out, &out_layout),
            buffer_offset(cond, cond_layout),
            buffer_offset(on_true, on_true_layout),
            buffer_offset(on_false, on_false_layout),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    let wg = 32u32;
    let grid = (elem_count as u32).div_ceil(wg);
    cond.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_entry_points() {
        assert_eq!(unary_entry_point("uexp"), Some("exp_f32"));
        assert_eq!(unary_entry_point("usin"), None);
    }

    #[test]
    fn binary_entry_points() {
        assert_eq!(binary_entry_point("badd"), Some("add_f32"));
        assert_eq!(binary_entry_point("bmul"), Some("mul_f32"));
    }

    #[test]
    fn matmul_entry_points() {
        assert_eq!(matmul_entry_point(MatMulKernel::TiledVecF32), "matmul_tiled_vec_f32");
        assert_eq!(matmul_entry_point(MatMulKernel::TiledVecF16), "matmul_tiled_vec_f16");
        assert_eq!(matmul_shader_source(MatMulKernel::TiledF16), MATMUL_TILED_F16);
    }

    #[test]
    fn reduce_entry_points() {
        assert_eq!(reduce_entry_point(ReduceOp::Sum), Some("reduce_sum_f32"));
        assert_eq!(reduce_entry_point(ReduceOp::Max), Some("reduce_max_f32"));
        assert_eq!(unary_entry_point("ugelu"), Some("gelu_f32"));
        assert_eq!(reduce_entry_point(ReduceOp::Min), Some("reduce_min_f32"));
    }

    #[test]
    fn broadcast_3d_stride_pattern() {
        let layout = Layout::contiguous(&Shape::from((1, 8, 1)))
            .broadcast_as(Shape::from((1, 8, 16)))
            .unwrap();
        assert_eq!(layout.stride(), &[8, 1, 0]);
    }

    #[test]
    fn materialize_broadcast_3d_layout() {
        let layout = Layout::contiguous(&Shape::from((1, 8, 1)))
            .broadcast_as(Shape::from((1, 8, 16)))
            .unwrap();
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0f32; 128];
        for (flat_i, src_i) in layout.strided_index().enumerate() {
            dst[flat_i] = src[src_i];
        }
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[15], 0.0);
        assert_eq!(dst[16], 1.0);
        assert_eq!(dst[31], 1.0);
        assert_eq!(dst[32], 2.0);
    }
}
