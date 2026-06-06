use super::bind_group::{ArgSortUniforms, BindGroupBuilder};
use super::kernel::WgpuKernel;
use super::storage::{buffer_offset, WgpuStorage};
use crate::wgsl::{ARGSORT, ARGSORT_BF16, ARGSORT_F16, ARGSORT_U32};
use crate::{DType, Error, Layout, Result as CandleResult};

/// Maximum last-dimension size supported by the GPU bitonic argsort kernel.
pub const MAX_ARGSORT_NCOLS_PAD: usize = 1024;

const ARGSORT_WG_SIZE: u32 = 256;

fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2;
    }
    n
}

fn argsort_shader(dtype: DType) -> Option<(&'static str, &'static str)> {
    match dtype {
        DType::F32 => Some((ARGSORT, "asort_asc_f32")),
        DType::F16 => Some((ARGSORT_F16, "asort_asc_f16")),
        DType::BF16 => Some((ARGSORT_BF16, "asort_asc_bf16")),
        DType::U32 => Some((ARGSORT_U32, "asort_asc_u32")),
        _ => None,
    }
}

fn argsort_entry(dtype: DType, asc: bool) -> Option<(&'static str, &'static str)> {
    let (shader, _) = argsort_shader(dtype)?;
    let entry = match (dtype, asc) {
        (DType::F32, true) => "asort_asc_f32",
        (DType::F32, false) => "asort_desc_f32",
        (DType::F16, true) => "asort_asc_f16",
        (DType::F16, false) => "asort_desc_f16",
        (DType::BF16, true) => "asort_asc_bf16",
        (DType::BF16, false) => "asort_desc_bf16",
        (DType::U32, true) => "asort_asc_u32",
        (DType::U32, false) => "asort_desc_u32",
        _ => return None,
    };
    Some((shader, entry))
}

/// Whether GPU argsort supports this dtype and last-dimension size.
pub fn gpu_argsort_supported(dtype: DType, last_dim: usize) -> bool {
    if argsort_shader(dtype).is_none() {
        return false;
    }
    let ncols_pad = next_power_of_2(last_dim);
    ncols_pad <= MAX_ARGSORT_NCOLS_PAD
}

/// Argsort along the last dimension; output is `u32` indices.
pub fn dispatch_arg_sort_last_dim(
    storage: &WgpuStorage,
    layout: &Layout,
    asc: bool,
) -> CandleResult<WgpuStorage> {
    if !layout.is_contiguous() {
        return Err(Error::RequiresContiguous {
            op: "arg_sort_last_dim",
        });
    }
    let dtype = storage.dtype();
    let last_dim = *layout
        .dims()
        .last()
        .ok_or_else(|| Error::Msg("empty last-dim in argsort".into()))?;
    if !gpu_argsort_supported(dtype, last_dim) {
        return Err(Error::UnsupportedDTypeForOp(dtype, "argsort").bt());
    }
    let ncols_pad = next_power_of_2(last_dim);
    let elem_count = layout.shape().elem_count();
    let nrows = elem_count / last_dim;
    let device = storage.device();
    let out = WgpuStorage::alloc(device, layout.shape(), DType::U32)?;
    let out_layout = Layout::contiguous(layout.shape());
    let uniforms = ArgSortUniforms {
        ncols: last_dim as u32,
        ncols_pad: ncols_pad as u32,
        asc: u32::from(asc),
        _pad: [0; 69],
    };
    let (shader, entry) = argsort_entry(dtype, asc)
        .ok_or_else(|| Error::UnsupportedDTypeForOp(dtype, "argsort").bt())?;
    let kernel = WgpuKernel::compile_with_workgroup_size(device, shader, entry, ARGSORT_WG_SIZE)?;
    let in0 = buffer_offset(storage, layout);
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&out, &out_layout),
        in0.clone(),
        Some(in0),
        uniforms.as_bytes(),
    )?;
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [nrows as u32, 1, 1]))?;
    Ok(out)
}
