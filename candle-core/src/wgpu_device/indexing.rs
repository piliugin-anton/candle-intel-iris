use super::bind_group::{BindGroupBuilder, IndexingUniforms};
use super::error::{Result, WgpuError};
use super::kernel::WgpuKernel;
use super::ops::dispatch_copy_strided_src;
use super::storage::{buffer_offset, WgpuStorage};
use crate::backend::BackendStorage;
use crate::wgsl::INDEXING;
use crate::{DType, Error, Layout, Result as CandleResult, Shape};

fn require_contiguous(layout: &Layout, op: &'static str) -> CandleResult<()> {
    if layout.contiguous_offsets().is_some() {
        Ok(())
    } else {
        Err(Error::RequiresContiguous { op }.bt())
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
        (DType::U32, "gather") => Some("gather_f32_u32"),
        (DType::U8, "gather") => Some("gather_f32_u8"),
        (DType::U32, "scatter") => Some("scatter_f32_u32"),
        (DType::U8, "scatter") => Some("scatter_f32_u8"),
        (DType::U32, "scatter_add") => Some("scatter_add_f32_u32"),
        (DType::U8, "scatter_add") => Some("scatter_add_f32_u8"),
        (DType::U32, "index_add") => Some("index_add_f32_u32"),
        (DType::U8, "index_add") => Some("index_add_f32_u8"),
        _ => None,
    }
}

fn gpu_indexing_supported(data_dtype: DType, ids_dtype: DType) -> bool {
    data_dtype == DType::F32 && indexing_entry(ids_dtype, "gather").is_some()
}

fn compile_indexing_kernel(device: &super::WgpuDevice, entry: &str) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(INDEXING, device.caps());
    WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, device.caps().elem_workgroup_size)
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
    let wg = device.caps().elem_workgroup_size;
    let grid = (elem_count as u32).div_ceil(wg);
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
    let src_cpu = src.to_cpu_storage().map_err(Error::from)?;
    let ids_cpu = ids.to_cpu_storage().map_err(Error::from)?;
    let out_cpu = src_cpu
        .gather(src_l, &ids_cpu, ids_l, dim)
        .map_err(Error::from)?;
    WgpuStorage::from_cpu(src.device(), &out_cpu).map_err(Error::from)
}

pub fn index_select_via_cpu(
    src: &WgpuStorage,
    src_l: &Layout,
    ids: &WgpuStorage,
    ids_l: &Layout,
    dim: usize,
) -> CandleResult<WgpuStorage> {
    let src_cpu = src.to_cpu_storage().map_err(Error::from)?;
    let ids_cpu = ids.to_cpu_storage().map_err(Error::from)?;
    let out_cpu = src_cpu
        .index_select(&ids_cpu, src_l, ids_l, dim)
        .map_err(Error::from)?;
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
    let dst_cpu = dst.to_cpu_storage().map_err(Error::from)?;
    let ids_cpu = ids.to_cpu_storage().map_err(Error::from)?;
    let src_cpu = src.to_cpu_storage().map_err(Error::from)?;
    let out_cpu = dst_cpu
        .index_add(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)
        .map_err(Error::from)?;
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
    let mut dst_cpu = dst.to_cpu_storage().map_err(Error::from)?;
    let ids_cpu = ids.to_cpu_storage().map_err(Error::from)?;
    let src_cpu = src.to_cpu_storage().map_err(Error::from)?;
    if add {
        dst_cpu
            .scatter_add_set(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)
            .map_err(Error::from)?;
    } else {
        dst_cpu
            .scatter_set(dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim)
            .map_err(Error::from)?;
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
    require_contiguous(src_l, "gather")?;
    require_contiguous(ids_l, "gather")?;
    let entry = indexing_entry(ids.dtype(), "gather")
        .ok_or_else(|| WgpuError::Message(format!("wgpu gather unsupported ids {:?}", ids.dtype())))?;
    let (left, src_dim, src_right) = split_dim(src_l.dims(), dim);
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
        src,
        src_l,
        ids,
        ids_l,
    )
    .map_err(Error::from)?;
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
    require_contiguous(src_l, "index-select")?;
    require_contiguous(ids_l, "index-select")?;
    let entry = indexing_entry(ids.dtype(), "index_select").ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu index_select unsupported ids {:?}",
            ids.dtype()
        ))
    })?;
    let (left, src_dim, right) = split_dim(src_l.dims(), dim);
    let ids_dim_size = ids_l.shape().elem_count();
    let elem_count = ids_dim_size * left * right;
    let mut out_dims = src_l.dims().to_vec();
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
        src,
        src_l,
        ids,
        ids_l,
    )
    .map_err(Error::from)?;
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
    require_contiguous(dst_l, "scatter")?;
    require_contiguous(ids_l, "scatter")?;
    require_contiguous(src_l, "scatter")?;
    let op = if add { "scatter_add" } else { "scatter" };
    let entry = indexing_entry(ids.dtype(), op)
        .ok_or_else(|| WgpuError::Message(format!("wgpu {op} unsupported ids {:?}", ids.dtype())))?;
    let (left, src_dim, right) = split_dim(src_l.dims(), dim);
    let dst_dim = dst_l.dims()[dim];
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
        dst.device(),
        entry,
        elem_count,
        uniforms,
        dst,
        dst_l,
        src,
        src_l,
        ids,
        ids_l,
    )
    .map_err(Error::from)?;
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
    require_contiguous(ids_l, "index-add")?;
    require_contiguous(src_l, "index-add")?;
    let entry = indexing_entry(ids.dtype(), "index_add").ok_or_else(|| {
        WgpuError::Message(format!("wgpu index_add unsupported ids {:?}", ids.dtype()))
    })?;
    let mut acc = WgpuStorage::alloc(dst.device(), dst_l.shape(), DType::F32).map_err(Error::from)?;
    dispatch_copy_strided_src(dst, &mut acc, 0, dst_l).map_err(Error::from)?;
    let (left, src_dim, right) = split_dim(src_l.dims(), dim);
    let dst_dim = dst_l.dims()[dim];
    let ids_dim_size = ids_l.dims()[0];
    let elem_count = left * right;
    let acc_layout = Layout::contiguous(dst_l.shape());
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
        src,
        src_l,
        ids,
        ids_l,
    )
    .map_err(Error::from)?;
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_device::WgpuDevice;

    #[test]
    fn indexing_shader_entry_points() {
        assert!(INDEXING.contains("fn index_select_f32_u32"));
        assert!(INDEXING.contains("fn gather_f32_u8"));
        assert!(INDEXING.contains("fn scatter_add_f32_u32"));
    }

    #[test]
    fn indexing_kernel_compiles_on_noop_device() {
        let device = WgpuDevice::new_test(false, 4096);
        compile_indexing_kernel(&device, "gather_f32_u32").expect("gather kernel compiles");
    }

    #[test]
    fn split_dim_sizes() {
        let dims = [2, 3, 4];
        assert_eq!(split_dim(&dims, 0), (1, 2, 12));
        assert_eq!(split_dim(&dims, 1), (2, 3, 4));
        assert_eq!(split_dim(&dims, 2), (6, 4, 1));
    }
}
