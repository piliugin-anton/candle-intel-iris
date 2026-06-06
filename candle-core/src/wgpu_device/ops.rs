use super::bind_group::{BindGroupBuilder, KernelUniforms, MatMulUniforms, ReduceUniforms};
use super::error::{Result, WgpuError};
use super::intel_caps::IntelGeneration;
use super::kernel::WgpuKernel;
use super::storage::{buffer_offset, WgpuStorage};
use super::WgpuDevice;
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, ReduceOp, UnaryOpT};
use crate::wgsl::{BINARY, MATMUL_NAIVE, MATMUL_TILED, MATMUL_WORKGROUP_SIZE, REDUCE, UNARY};
use crate::{CpuStorage, DType, Error, Layout, Result as CandleResult, Shape};

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

fn unary_entry_point(kernel: &str) -> Option<&'static str> {
    match kernel {
        "uexp" => Some("exp_f32"),
        "uneg" => Some("neg_f32"),
        "ulog" => Some("log_f32"),
        "usqrt" => Some("sqrt_f32"),
        "urelu" => Some("relu_f32"),
        "urecip" => Some("recip_f32"),
        "usilu" => Some("silu_f32"),
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
        ReduceOp::ArgMax => Some("reduce_argmax_f32"),
        ReduceOp::Min | ReduceOp::ArgMin => None,
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

fn compile_matmul_kernel(device: &WgpuDevice, tiled: bool) -> Result<WgpuKernel> {
    let source = if tiled { MATMUL_TILED } else { MATMUL_NAIVE };
    let entry = if tiled {
        "matmul_tiled_f32"
    } else {
        "matmul_naive_f32"
    };
    WgpuKernel::compile_with_workgroup_size(device, source, entry, MATMUL_WORKGROUP_SIZE)
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
    (b, m, n, _k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(lhs.dtype(), "matmul")?;
    require_f32(rhs.dtype(), "matmul")?;

    let device = lhs.device().clone();
    let lhs_dims = lhs_layout.shape().dims();
    let dim = lhs_dims.len();
    let mut out_dims = lhs_dims[..dim - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);
    let out_layout = Layout::contiguous(&out_shape);
    let _elem_count = b * m * n;
    let out = WgpuStorage::alloc(&device, &out_shape, DType::F32)?;

    let uniforms = MatMulUniforms::new(b, m, n, _k, &out_layout, lhs_layout, rhs_layout);
    let tiled = matches!(
        device.caps().generation,
        IntelGeneration::Gen12Plus | IntelGeneration::Gen11
    );
    let kernel = compile_matmul_kernel(&device, tiled)?;
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

    let wg = MATMUL_WORKGROUP_SIZE;
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

pub fn dispatch_reduce(
    storage: &WgpuStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), "reduce")?;

    if matches!(op, ReduceOp::Min | ReduceOp::ArgMin) {
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
    fn reduce_entry_points() {
        assert_eq!(reduce_entry_point(ReduceOp::Sum), Some("reduce_sum_f32"));
        assert_eq!(reduce_entry_point(ReduceOp::Max), Some("reduce_max_f32"));
        assert_eq!(reduce_entry_point(ReduceOp::Min), None);
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
