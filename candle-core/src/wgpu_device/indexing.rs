use super::bind_group::{BindGroupBuilder, IndexingUniforms};
use super::error::{Result, WgpuError};
use super::kernel::{elemwise_workgroup_count, WgpuKernel};
use super::ops::{dispatch_copy_strided_src, dispatch_copy_strided_u32};
use super::storage::{buffer_offset, WgpuStorage};
use crate::backend::BackendStorage;
use crate::wgsl::INDEXING;
use crate::{DType, Error, Layout, Result as CandleResult, Shape};

/// Materialize broadcast/strided float activations on GPU before indexing kernels run.
fn ensure_contiguous_float(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<(WgpuStorage, Layout)> {
    if layout.is_contiguous() {
        return Ok((storage.clone(), layout.clone()));
    }
    let out = WgpuStorage::alloc(storage.device(), layout.shape(), storage.dtype())
        .map_err(Error::from)?;
    let out_layout = Layout::contiguous(layout.shape());
    let mut out_mut = out.clone();
    dispatch_copy_strided_src(storage, &mut out_mut, 0, layout).map_err(Error::from)?;
    Ok((out, out_layout))
}

/// Materialize broadcast/strided index tensors (u32/i32) on GPU — common in MoE routing.
fn ensure_contiguous_ids(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<(WgpuStorage, Layout)> {
    if layout.is_contiguous() {
        return Ok((storage.clone(), layout.clone()));
    }
    match storage.dtype() {
        DType::U32 | DType::I32 => {
            let out = WgpuStorage::alloc(storage.device(), layout.shape(), storage.dtype())
                .map_err(Error::from)?;
            let out_layout = Layout::contiguous(layout.shape());
            let mut out_mut = out.clone();
            dispatch_copy_strided_u32(storage, &mut out_mut, 0, layout).map_err(Error::from)?;
            Ok((out, out_layout))
        }
        // U8 indices are byte-packed; uncommon strided layouts use the CPU path upstream.
        _ => Err(Error::RequiresContiguous { op: "indexing-ids" }.bt()),
    }
}

fn split_dim(dims: &[usize], dim: usize) -> (usize, usize, usize) {
    let left = dims[..dim].iter().product();
    let mid = dims[dim];
    let right = dims[dim + 1..].iter().product();
    (left, mid, right)
}

fn indexing_entry(ids_dtype: DType, op: &str) -> Option<&'static str> {
    match (ids_dtype, op) {
        (DType::U32, "index_select") => Some("index_select_f32_u32"),
        (DType::U8, "index_select") => Some("index_select_f32_u8"),
        (DType::I32, "index_select") => Some("index_select_f32_i32"),
        (DType::U32, "gather") => Some("gather_f32_u32"),
        (DType::U8, "gather") => Some("gather_f32_u8"),
        (DType::I32, "gather") => Some("gather_f32_i32"),
        (DType::U32, "scatter") => Some("scatter_f32_u32"),
        (DType::U8, "scatter") => Some("scatter_f32_u8"),
        (DType::I32, "scatter") => Some("scatter_f32_i32"),
        (DType::U32, "scatter_add") => Some("scatter_add_f32_u32"),
        (DType::U8, "scatter_add") => Some("scatter_add_f32_u8"),
        (DType::I32, "scatter_add") => Some("scatter_add_f32_i32"),
        (DType::U32, "index_add") => Some("index_add_f32_u32"),
        (DType::U8, "index_add") => Some("index_add_f32_u8"),
        (DType::I32, "index_add") => Some("index_add_f32_i32"),
        _ => None,
    }
}

fn gpu_indexing_data_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

fn gpu_indexing_supported(data_dtype: DType, ids_dtype: DType) -> bool {
    gpu_indexing_data_dtype(data_dtype) && indexing_entry(ids_dtype, "gather").is_some()
}

/// Cast low-precision activations to f32 for the indexing kernels; returns `(storage, layout, restore_dtype)`.
fn indexing_data_as_f32(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<(WgpuStorage, Layout, Option<DType>)> {
    match storage.dtype() {
        DType::F32 => Ok((storage.clone(), layout.clone(), None)),
        DType::F16 | DType::BF16 => {
            let f32 = storage.to_dtype(layout, DType::F32)?;
            let f32_layout = Layout::contiguous(layout.shape());
            Ok((f32, f32_layout, Some(storage.dtype())))
        }
        other => Err(Error::UnsupportedDTypeForOp(other, "indexing").bt()),
    }
}

fn compile_indexing_kernel(device: &super::WgpuDevice, entry: &str) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(INDEXING, device.caps());
    WgpuKernel::compile_with_workgroup_size(
        device,
        &tuned,
        entry,
        device.caps().elem_workgroup_size,
    )
}

fn dispatch_indexing(
    device: &super::WgpuDevice,
    entry: &str,
    elem_count: usize,
    uniforms: IndexingUniforms,
    output: &WgpuStorage,
    out_layout: &Layout,
    input: &WgpuStorage,
    in_layout: &Layout,
    ids: &WgpuStorage,
    ids_layout: &Layout,
) -> Result<()> {
    let kernel = compile_indexing_kernel(device, entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(output, out_layout),
        buffer_offset(input, in_layout),
        Some(buffer_offset(ids, ids_layout)),
        uniforms.as_bytes(),
    )?;
    let grid = elemwise_workgroup_count(device, elem_count);
    output
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

pub fn gather_via_cpu(
    src: &WgpuStorage,
    src_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    let src_cpu = src.to_cpu_storage()?;
    let ids_cpu = ids.to_cpu_storage()?;
    let out_cpu = src_cpu.gather(src_l, &ids_cpu, ids_l, dim)?;
    WgpuStorage::from_cpu(src.device(), &out_cpu).map_err(Error::from)
}

pub fn index_select_via_cpu(
    src: &WgpuStorage,
    src_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    let src_cpu = src.to_cpu_storage()?;
    let ids_cpu = ids.to_cpu_storage()?;
    let out_cpu = src_cpu.index_select(&ids_cpu, src_l, ids_l, dim)?;
    WgpuStorage::from_cpu(src.device(), &out_cpu).map_err(Error::from)
}

pub fn index_add_via_cpu(
    dst: &WgpuStorage,
    dst_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    src: &WgpuStorage,
    src_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    let dst_cpu = dst.to_cpu_storage()?;
    let ids_cpu = ids.to_cpu_storage()?;
    let src_cpu = src.to_cpu_storage()?;
    let out_cpu = dst_cpu.index_add(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
    WgpuStorage::from_cpu(dst.device(), &out_cpu).map_err(Error::from)
}

pub fn scatter_via_cpu(
    dst: &mut WgpuStorage,
    dst_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    src: &WgpuStorage,
    src_l: &Layout,
    dim: usize,
    add: bool,
) -> CandleResult<()> {
    let mut dst_cpu = dst.to_cpu_storage()?;
    let ids_cpu = ids.to_cpu_storage()?;
    let src_cpu = src.to_cpu_storage()?;
    if add {
        dst_cpu.scatter_add_set(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
    } else {
        dst_cpu.scatter_set(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
    }
    *dst = WgpuStorage::from_cpu(dst.device(), &dst_cpu).map_err(Error::from)?;
    Ok(())
}

pub fn dispatch_gather_f32(
    src: &WgpuStorage,
    src_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    if !gpu_indexing_supported(src.dtype(), ids.dtype()) {
        return gather_via_cpu(src, src_l, ids, ids_l, dim);
    }
    let (src_f32, src_f32_l, restore_dtype) = indexing_data_as_f32(src, src_l)?;
    let (src_f32, src_f32_l) = match ensure_contiguous_float(&src_f32, &src_f32_l) {
        Ok(v) => v,
        Err(_) => return gather_via_cpu(src, src_l, ids, ids_l, dim),
    };
    let (ids, ids_l) = match ensure_contiguous_ids(ids, ids_l) {
        Ok(v) => v,
        Err(_) => return gather_via_cpu(src, src_l, ids, ids_l, dim),
    };
    let entry = indexing_entry(ids.dtype(), "gather").ok_or_else(|| {
        WgpuError::Message(format!("wgpu gather unsupported ids {:?}", ids.dtype()))
    })?;
    let (left, src_dim, src_right) = split_dim(src_f32_l.dims(), dim);
    let ids_dim = ids_l.dims()[dim];
    let ids_right: usize = ids_l.dims()[dim + 1..].iter().product();
    let elem_count = ids_l.shape().elem_count();
    let out_shape = ids_l.shape().clone();
    let out = WgpuStorage::alloc(src.device(), &out_shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(&out_shape);
    let uniforms = IndexingUniforms {
        elem_count: elem_count as u32,
        left_size: left as u32,
        src_dim_size: src_dim as u32,
        dim_size: ids_dim as u32,
        right_size: ids_right as u32,
        ids_dim_size: src_right as u32,
        _pad: [0; 66],
    };
    dispatch_indexing(
        src.device(),
        entry,
        elem_count,
        uniforms,
        &out,
        &out_layout,
        &src_f32,
        &src_f32_l,
        &ids,
        &ids_l,
    )
    .map_err(Error::from)?;
    if let Some(dtype) = restore_dtype {
        return out.to_dtype(&out_layout, dtype);
    }
    Ok(out)
}

pub fn dispatch_index_select_f32(
    src: &WgpuStorage,
    src_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    if !gpu_indexing_supported(src.dtype(), ids.dtype()) {
        return index_select_via_cpu(src, src_l, ids, ids_l, dim);
    }
    let (src_f32, src_f32_l, restore_dtype) = indexing_data_as_f32(src, src_l)?;
    let (src_f32, src_f32_l) = match ensure_contiguous_float(&src_f32, &src_f32_l) {
        Ok(v) => v,
        Err(_) => return index_select_via_cpu(src, src_l, ids, ids_l, dim),
    };
    let (ids, ids_l) = match ensure_contiguous_ids(ids, ids_l) {
        Ok(v) => v,
        Err(_) => return index_select_via_cpu(src, src_l, ids, ids_l, dim),
    };
    let entry = indexing_entry(ids.dtype(), "index_select").ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu index_select unsupported ids {:?}",
            ids.dtype()
        ))
    })?;
    let (left, src_dim, right) = split_dim(src_f32_l.dims(), dim);
    let ids_dim_size = ids_l.shape().elem_count();
    let elem_count = ids_dim_size * left * right;
    let mut out_dims = src_f32_l.dims().to_vec();
    out_dims[dim] = ids_l.dims()[0];
    let out_shape = Shape::from_dims(&out_dims);
    let out = WgpuStorage::alloc(src.device(), &out_shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(&out_shape);
    let uniforms = IndexingUniforms {
        elem_count: elem_count as u32,
        left_size: left as u32,
        src_dim_size: src_dim as u32,
        dim_size: ids_dim_size as u32,
        right_size: right as u32,
        ids_dim_size: 0,
        _pad: [0; 66],
    };
    dispatch_indexing(
        src.device(),
        entry,
        elem_count,
        uniforms,
        &out,
        &out_layout,
        &src_f32,
        &src_f32_l,
        &ids,
        &ids_l,
    )
    .map_err(Error::from)?;
    if let Some(dtype) = restore_dtype {
        return out.to_dtype(&out_layout, dtype);
    }
    Ok(out)
}

pub fn dispatch_scatter_f32(
    dst: &mut WgpuStorage,
    dst_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    src: &WgpuStorage,
    src_l: &Layout,
    dim: usize,
    add: bool,
) -> CandleResult<()> {
    if !gpu_indexing_supported(dst.dtype(), ids.dtype()) {
        return scatter_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim, add);
    }
    let (src_f32, src_f32_l, _) = indexing_data_as_f32(src, src_l)?;
    let (src_f32, src_f32_l) = match ensure_contiguous_float(&src_f32, &src_f32_l) {
        Ok(v) => v,
        Err(_) => return scatter_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim, add),
    };
    let (ids, ids_l) = match ensure_contiguous_ids(ids, ids_l) {
        Ok(v) => v,
        Err(_) => return scatter_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim, add),
    };
    let op = if add { "scatter_add" } else { "scatter" };
    let entry = indexing_entry(ids.dtype(), op).ok_or_else(|| {
        WgpuError::Message(format!("wgpu {op} unsupported ids {:?}", ids.dtype()))
    })?;
    let (left, src_dim, right) = split_dim(src_f32_l.dims(), dim);
    let device = dst.device().clone();

    let run = |dst_buf: &mut WgpuStorage, dst_buf_l: &Layout| -> CandleResult<()> {
        let dst_dim = dst_buf_l.dims()[dim];
        let elem_count = left * right;
        let uniforms = IndexingUniforms {
            elem_count: elem_count as u32,
            left_size: left as u32,
            src_dim_size: src_dim as u32,
            dim_size: dst_dim as u32,
            right_size: right as u32,
            ids_dim_size: 0,
            _pad: [0; 66],
        };
        dispatch_indexing(
            &device, entry, elem_count, uniforms, dst_buf, dst_buf_l, &src_f32, &src_f32_l, &ids,
            &ids_l,
        )
        .map_err(Error::from)
    };

    if !dst_l.is_contiguous() {
        return scatter_via_cpu(dst, dst_l, &ids, &ids_l, src, src_l, dim, add);
    }
    if matches!(dst.dtype(), DType::F16 | DType::BF16) {
        let dtype = dst.dtype();
        let mut dst_f32 = dst.to_dtype(dst_l, DType::F32)?;
        let dst_f32_l = Layout::contiguous(dst_l.shape());
        run(&mut dst_f32, &dst_f32_l)?;
        *dst = dst_f32.to_dtype(&dst_f32_l, dtype)?;
    } else {
        run(dst, dst_l)?;
    }
    Ok(())
}

pub fn dispatch_index_add_f32(
    dst: &WgpuStorage,
    dst_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    src: &WgpuStorage,
    src_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    if !gpu_indexing_supported(dst.dtype(), ids.dtype()) {
        return index_add_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim);
    }
    let (dst_f32, dst_f32_l, dst_restore) = indexing_data_as_f32(dst, dst_l)?;
    let (src_f32, src_f32_l, _) = indexing_data_as_f32(src, src_l)?;
    let (src_f32, src_f32_l) = match ensure_contiguous_float(&src_f32, &src_f32_l) {
        Ok(v) => v,
        Err(_) => return index_add_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim),
    };
    let (ids, ids_l) = match ensure_contiguous_ids(ids, ids_l) {
        Ok(v) => v,
        Err(_) => return index_add_via_cpu(dst, dst_l, ids, ids_l, src, src_l, dim),
    };
    let (dst_f32, dst_f32_l) = match ensure_contiguous_float(&dst_f32, &dst_f32_l) {
        Ok(v) => v,
        Err(_) => return index_add_via_cpu(dst, dst_l, &ids, &ids_l, src, src_l, dim),
    };
    let entry = indexing_entry(ids.dtype(), "index_add").ok_or_else(|| {
        WgpuError::Message(format!("wgpu index_add unsupported ids {:?}", ids.dtype()))
    })?;
    let mut acc =
        WgpuStorage::alloc(dst.device(), dst_f32_l.shape(), DType::F32).map_err(Error::from)?;
    dispatch_copy_strided_src(&dst_f32, &mut acc, 0, &dst_f32_l).map_err(Error::from)?;
    let (left, src_dim, right) = split_dim(src_f32_l.dims(), dim);
    let dst_dim = dst_f32_l.dims()[dim];
    let ids_dim_size = ids_l.dims()[0];
    let elem_count = left * right;
    let acc_layout = Layout::contiguous(dst_f32_l.shape());
    let uniforms = IndexingUniforms {
        elem_count: elem_count as u32,
        left_size: left as u32,
        src_dim_size: src_dim as u32,
        dim_size: dst_dim as u32,
        right_size: right as u32,
        ids_dim_size: ids_dim_size as u32,
        _pad: [0; 66],
    };
    dispatch_indexing(
        dst.device(),
        entry,
        elem_count,
        uniforms,
        &acc,
        &acc_layout,
        &src_f32,
        &src_f32_l,
        &ids,
        &ids_l,
    )
    .map_err(Error::from)?;
    if let Some(dtype) = dst_restore {
        return acc.to_dtype(&acc_layout, dtype);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_device::WgpuDevice;

    #[test]
    fn indexing_shader_entry_points() {
        assert!(INDEXING.contains("fn index_select_f32_u32"));
        assert!(INDEXING.contains("fn index_select_f32_i32"));
        assert!(INDEXING.contains("fn gather_f32_u8"));
        assert!(INDEXING.contains("fn scatter_add_f32_u32"));
    }

    #[test]
    fn indexing_kernel_compiles_on_noop_device() {
        let device = WgpuDevice::new_test(false, 4096);
        compile_indexing_kernel(&device, "gather_f32_u32").expect("gather kernel compiles");
        compile_indexing_kernel(&device, "gather_f32_i32").expect("gather i32 kernel compiles");
    }

    #[test]
    fn copy_strided_u32_kernel_compiles_on_noop_device() {
        use super::super::kernel::WgpuKernel;
        use crate::wgsl::COPY_U32;
        let device = WgpuDevice::new_test(false, 4096);
        WgpuKernel::compile_with_workgroup_size(&device, COPY_U32, "copy_strided_u32", 32)
            .expect("copy_strided_u32 compiles");
    }

    #[test]
    fn split_dim_sizes() {
        let dims = [2, 3, 4];
        assert_eq!(split_dim(&dims, 0), (1, 2, 12));
        assert_eq!(split_dim(&dims, 1), (2, 3, 4));
        assert_eq!(split_dim(&dims, 2), (6, 4, 1));
    }
}
