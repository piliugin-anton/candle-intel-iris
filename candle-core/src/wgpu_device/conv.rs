use super::bind_group::{
    BindGroupBuilder, ConvTranspose1dUniforms, ConvTranspose2dUniforms, Im2col1dUniforms,
    Im2col2dUniforms, NhwcToNchwUniforms, Pool2dUniforms, TensorLayoutUniform,
    UpsampleBilinear2dUniforms, UpsampleNearest1dUniforms, UpsampleNearest2dUniforms,
};
use super::error::{Result, WgpuError};
use super::intel_caps::tune_shader_source;
use super::kernel::WgpuKernel;
use super::ops::{dispatch_copy_strided_src, dispatch_matmul, float_type_suffix, require_float};
use super::storage::{buffer_offset, WgpuStorage};
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::wgsl::{
    CONV_TRANSPOSE1D, CONV_TRANSPOSE1D_BF16, CONV_TRANSPOSE1D_F16, CONV_TRANSPOSE2D,
    CONV_TRANSPOSE2D_BF16, CONV_TRANSPOSE2D_F16, IM2COL1D, IM2COL1D_BF16, IM2COL1D_F16, IM2COL2D,
    IM2COL2D_BF16, IM2COL2D_F16, NHWC_TO_NCHW, NHWC_TO_NCHW_BF16, NHWC_TO_NCHW_F16, POOL2D,
    POOL2D_BF16, POOL2D_F16, UPSAMPLE_BILINEAR2D, UPSAMPLE_BILINEAR2D_BF16, UPSAMPLE_BILINEAR2D_F16,
    UPSAMPLE_NEAREST1D, UPSAMPLE_NEAREST1D_BF16, UPSAMPLE_NEAREST1D_F16, UPSAMPLE_NEAREST2D,
    UPSAMPLE_NEAREST2D_BF16, UPSAMPLE_NEAREST2D_F16,
};
use crate::{DType, Error, Layout, Result as CandleResult, Shape};

fn im2col2d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => IM2COL2D_F16,
        DType::BF16 => IM2COL2D_BF16,
        _ => IM2COL2D,
    }
}

fn nhwc_to_nchw_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => NHWC_TO_NCHW_F16,
        DType::BF16 => NHWC_TO_NCHW_BF16,
        _ => NHWC_TO_NCHW,
    }
}

fn im2col1d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => IM2COL1D_F16,
        DType::BF16 => IM2COL1D_BF16,
        _ => IM2COL1D,
    }
}

fn pool2d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => POOL2D_F16,
        DType::BF16 => POOL2D_BF16,
        _ => POOL2D,
    }
}

fn upsample_nearest1d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => UPSAMPLE_NEAREST1D_F16,
        DType::BF16 => UPSAMPLE_NEAREST1D_BF16,
        _ => UPSAMPLE_NEAREST1D,
    }
}

fn upsample_nearest2d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => UPSAMPLE_NEAREST2D_F16,
        DType::BF16 => UPSAMPLE_NEAREST2D_BF16,
        _ => UPSAMPLE_NEAREST2D,
    }
}

fn upsample_bilinear2d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => UPSAMPLE_BILINEAR2D_F16,
        DType::BF16 => UPSAMPLE_BILINEAR2D_BF16,
        _ => UPSAMPLE_BILINEAR2D,
    }
}

fn conv_transpose1d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => CONV_TRANSPOSE1D_F16,
        DType::BF16 => CONV_TRANSPOSE1D_BF16,
        _ => CONV_TRANSPOSE1D,
    }
}

fn conv_transpose2d_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => CONV_TRANSPOSE2D_F16,
        DType::BF16 => CONV_TRANSPOSE2D_BF16,
        _ => CONV_TRANSPOSE2D,
    }
}

fn with_compute_dtype<F>(
    storage: &WgpuStorage,
    layout: &Layout,
    out_shape: &Shape,
    op: &'static str,
    f: F,
) -> CandleResult<WgpuStorage>
where
    F: Fn(&WgpuStorage, &Layout, DType) -> CandleResult<WgpuStorage>,
{
    require_float(storage.dtype(), op)?;
    let device = storage.device();
    let storage_dtype = storage.dtype();
    let compute_dtype = if device.caps().has_gpu_kernels_for(storage_dtype) {
        storage_dtype
    } else {
        device.effective_dtype(storage_dtype)
    };
    if compute_dtype != storage_dtype {
        let cast = storage.to_dtype(layout, compute_dtype)?;
        let cast_layout = Layout::contiguous(layout.shape());
        let out = f(&cast, &cast_layout, compute_dtype)?;
        return out.to_dtype(&Layout::contiguous(out_shape), storage_dtype);
    }
    f(storage, layout, compute_dtype)
}

fn workgroup_count(device: &WgpuDevice, elem_count: usize) -> u32 {
    super::kernel::elemwise_workgroup_count(device, elem_count)
}

fn compile_standard_kernel(
    device: &WgpuDevice,
    source: &str,
    entry_point: &str,
) -> Result<WgpuKernel> {
    let tuned = tune_shader_source(source, device.caps());
    WgpuKernel::compile_with_workgroup_size(
        device,
        &tuned,
        entry_point,
        device.caps().elem_workgroup_size,
    )
}

fn ensure_contiguous(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<(WgpuStorage, Layout)> {
    if layout.is_contiguous() {
        return Ok((storage.clone(), layout.clone()));
    }
    let out = WgpuStorage::alloc(storage.device(), layout.shape(), storage.dtype())?;
    let out_layout = Layout::contiguous(layout.shape());
    let mut out_mut = out.clone();
    dispatch_copy_strided_src(storage, &mut out_mut, 0, layout)?;
    Ok((out, out_layout))
}

fn dispatch_im2col2d(
    device: &WgpuDevice,
    src: &WgpuStorage,
    src_layout: &Layout,
    h_out: usize,
    w_out: usize,
    h_k: usize,
    w_k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    dtype: DType,
) -> Result<WgpuStorage> {
    let dst_numel =
        src_layout.shape().dims()[0] * h_out * w_out * src_layout.shape().dims()[1] * h_k * w_k;
    let col = WgpuStorage::alloc(device, &Shape::from(dst_numel), dtype)?;
    let col_layout = Layout::contiguous(Shape::from(dst_numel));
    let uniforms = Im2col2dUniforms {
        dst_numel: dst_numel as u32,
        h_out: h_out as u32,
        w_out: w_out as u32,
        h_k: h_k as u32,
        w_k: w_k as u32,
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
        src_layout: TensorLayoutUniform::from_layout(src_layout),
        _pad: [0; 44],
    };
    let entry = format!("im2col2d_{}", float_type_suffix(dtype));
    let kernel = compile_standard_kernel(device, im2col2d_shader(dtype), &entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&col, &col_layout),
        buffer_offset(src, src_layout),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = workgroup_count(device, dst_numel);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(col)
}

fn dispatch_nhwc_to_nchw(
    device: &WgpuDevice,
    src: &WgpuStorage,
    src_offset: usize,
    dst: &WgpuStorage,
    dst_offset: usize,
    b: usize,
    h: usize,
    w: usize,
    c: usize,
    dtype: DType,
) -> Result<()> {
    let elem_count = b * h * w * c;
    let uniforms = NhwcToNchwUniforms {
        elem_count: elem_count as u32,
        b: b as u32,
        h: h as u32,
        w: w as u32,
        c: c as u32,
        src_offset: src_offset as u32,
        dst_offset: dst_offset as u32,
        _pad: [0; 65],
    };
    let entry = format!("nhwc_to_nchw_{}", float_type_suffix(dtype));
    let kernel = compile_standard_kernel(device, nhwc_to_nchw_shader(dtype), &entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(dst, &Layout::contiguous(Shape::from(elem_count))),
        buffer_offset(src, &Layout::contiguous_with_offset((b, h, w, c), src_offset)),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = workgroup_count(device, elem_count);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

fn dispatch_im2col1d(
    device: &WgpuDevice,
    src: &WgpuStorage,
    src_layout: &Layout,
    l_out: usize,
    l_k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    dtype: DType,
) -> Result<WgpuStorage> {
    let b = src_layout.shape().dims()[0];
    let c_in = src_layout.shape().dims()[1];
    let dst_numel = b * l_out * c_in * l_k;
    let col = WgpuStorage::alloc(device, &Shape::from(dst_numel), dtype)?;
    let col_layout = Layout::contiguous(Shape::from(dst_numel));
    let uniforms = Im2col1dUniforms {
        dst_numel: dst_numel as u32,
        l_out: l_out as u32,
        l_k: l_k as u32,
        stride: stride as u32,
        padding: padding as u32,
        dilation: dilation as u32,
        _align: [0; 2],
        src_layout: TensorLayoutUniform::from_layout(src_layout),
        _pad: [0; 44],
    };
    let entry = format!("im2col1d_{}", float_type_suffix(dtype));
    let kernel = compile_standard_kernel(device, im2col1d_shader(dtype), &entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&col, &col_layout),
        buffer_offset(src, src_layout),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = workgroup_count(device, dst_numel);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(col)
}

pub fn conv2d(
    self_: &WgpuStorage,
    layout: &Layout,
    kernel: &WgpuStorage,
    kernel_l: &Layout,
    params: &ParamsConv2D,
) -> CandleResult<WgpuStorage> {
    require_float(self_.dtype(), "conv2d")?;
    require_float(kernel.dtype(), "conv2d")?;
    if self_.dtype() != kernel.dtype() {
        return Err(Error::Msg("conv2d input and kernel dtype mismatch".into()));
    }

    let device = self_.device();
    let storage_dtype = self_.dtype();
    let compute_dtype = if device.caps().has_gpu_kernels_for(storage_dtype) {
        storage_dtype
    } else {
        device.effective_dtype(storage_dtype)
    };
    if compute_dtype != storage_dtype {
        let self_cast = self_.to_dtype(layout, compute_dtype)?;
        let kernel_cast = kernel.to_dtype(kernel_l, compute_dtype)?;
        let layout_cast = Layout::contiguous(layout.shape());
        let kernel_layout_cast = Layout::contiguous(kernel_l.shape());
        let out = conv2d(
            &self_cast,
            &layout_cast,
            &kernel_cast,
            &kernel_layout_cast,
            params,
        )?;
        let out_shape = Shape::from((params.b_size, params.c_out, params.out_h(), params.out_w()));
        return out.to_dtype(&Layout::contiguous(&out_shape), storage_dtype);
    }

    let h_out = params.out_h();
    let w_out = params.out_w();
    let col = dispatch_im2col2d(
        device,
        self_,
        layout,
        h_out,
        w_out,
        params.k_h,
        params.k_w,
        params.stride,
        params.padding,
        params.dilation,
        compute_dtype,
    )
    .map_err(Error::from)?;

    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_h * params.k_w * params.c_in;
    let m = h_out * w_out;
    let col_l = Layout::contiguous((b, m, k));

    let (kernel_storage, kernel_layout) = if kernel_l.is_contiguous() {
        (kernel.clone(), kernel_l.clone())
    } else {
        ensure_contiguous(kernel, kernel_l)?
    };

    let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_layout.start_offset())
        .transpose(1, 2)?
        .broadcast_as((b, k, n))?;

    let res = dispatch_matmul(&col, &kernel_storage, (b, m, n, k), &col_l, &kernel_l)?;

    let out_shape = Shape::from((b, n, h_out, w_out));
    let res_t = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
    dispatch_nhwc_to_nchw(device, &res, 0, &res_t, 0, b, h_out, w_out, n, compute_dtype)
        .map_err(Error::from)?;
    Ok(res_t)
}

pub fn conv1d(
    self_: &WgpuStorage,
    layout: &Layout,
    kernel: &WgpuStorage,
    kernel_l: &Layout,
    params: &ParamsConv1D,
) -> CandleResult<WgpuStorage> {
    require_float(self_.dtype(), "conv1d")?;
    require_float(kernel.dtype(), "conv1d")?;
    if self_.dtype() != kernel.dtype() {
        return Err(Error::Msg("conv1d input and kernel dtype mismatch".into()));
    }

    let device = self_.device();
    let storage_dtype = self_.dtype();
    let compute_dtype = if device.caps().has_gpu_kernels_for(storage_dtype) {
        storage_dtype
    } else {
        device.effective_dtype(storage_dtype)
    };
    if compute_dtype != storage_dtype {
        let self_cast = self_.to_dtype(layout, compute_dtype)?;
        let kernel_cast = kernel.to_dtype(kernel_l, compute_dtype)?;
        let layout_cast = Layout::contiguous(layout.shape());
        let kernel_layout_cast = Layout::contiguous(kernel_l.shape());
        let out = conv1d(
            &self_cast,
            &layout_cast,
            &kernel_cast,
            &kernel_layout_cast,
            params,
        )?;
        let out_shape = Shape::from((params.b_size, params.c_out, params.l_out()));
        return out.to_dtype(&Layout::contiguous(&out_shape), storage_dtype);
    }

    let l_out = params.l_out();
    let col = dispatch_im2col1d(
        device,
        self_,
        layout,
        l_out,
        params.k_size,
        params.stride,
        params.padding,
        params.dilation,
        compute_dtype,
    )
    .map_err(Error::from)?;

    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_size * params.c_in;
    let m = l_out;
    let col_l = Layout::contiguous((b, m, k));

    let (kernel_storage, kernel_layout) = if kernel_l.is_contiguous() {
        (kernel.clone(), kernel_l.clone())
    } else {
        ensure_contiguous(kernel, kernel_l)?
    };

    let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_layout.start_offset())
        .transpose(1, 2)?
        .broadcast_as((b, k, n))?;

    let res = dispatch_matmul(&col, &kernel_storage, (b, m, n, k), &col_l, &kernel_l)?;

    let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
    let mut res_t = WgpuStorage::alloc(device, res_l.shape(), compute_dtype)?;
    res.copy_strided_src(&mut res_t, 0, &res_l)?;
    Ok(res_t)
}

pub fn conv_transpose2d(
    self_: &WgpuStorage,
    layout: &Layout,
    kernel: &WgpuStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose2D,
) -> CandleResult<WgpuStorage> {
    with_compute_dtype(
        self_,
        layout,
        &Shape::from((params.b_size, params.c_out, params.out_h(), params.out_w())),
        "conv_transpose2d",
        |self_, layout, compute_dtype| {
            require_float(kernel.dtype(), "conv_transpose2d")?;
            if self_.dtype() != kernel.dtype() {
                return Err(Error::Msg(
                    "conv_transpose2d input and kernel dtype mismatch".into(),
                ));
            }

            let out_w = params.out_w();
            let out_h = params.out_h();
            let dst_el = params.c_out * out_w * out_h * params.b_size;
            let out_shape = Shape::from((params.b_size, params.c_out, out_h, out_w));
            let device = self_.device();
            let out = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
            let out_layout = Layout::contiguous(&out_shape);

            let (src, src_layout) = ensure_contiguous(self_, layout)?;
            let (kernel_storage, kernel_layout) = ensure_contiguous(kernel, kernel_l)?;

            let uniforms = ConvTranspose2dUniforms {
                w_out: out_w as u32,
                h_out: out_h as u32,
                stride: params.stride as u32,
                padding: params.padding as u32,
                output_padding: params.output_padding as u32,
                dilation: params.dilation as u32,
                dst_numel: dst_el as u32,
                _align: 0,
                src_layout: TensorLayoutUniform::from_layout(&src_layout),
                kernel_layout: TensorLayoutUniform::from_layout(&kernel_layout),
                _pad: [0; 24],
            };

            let entry = format!("conv_transpose2d_{}", float_type_suffix(compute_dtype));
            let wgpu_kernel = WgpuKernel::compile_extended(
                device,
                conv_transpose2d_shader(compute_dtype),
                &entry,
                32,
            )
            .map_err(Error::from)?;
            let k_off = buffer_offset(&kernel_storage, &kernel_layout);
            let bind_group = wgpu_kernel
                .create_extended_bind_group(
                    device,
                    buffer_offset(&out, &out_layout),
                    buffer_offset(&src, &src_layout),
                    k_off.clone(),
                    k_off,
                    uniforms.as_bytes(),
                )
                .map_err(Error::from)?;
            let grid = workgroup_count(device, dst_el);
            self_
                .backing()
                .with_unmapped(|| {
                    wgpu_kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1])
                })
                .map_err(Error::from)?;
            Ok(out)
        },
    )
}

pub fn conv_transpose1d(
    self_: &WgpuStorage,
    layout: &Layout,
    kernel: &WgpuStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose1D,
) -> CandleResult<WgpuStorage> {
    with_compute_dtype(
        self_,
        layout,
        &Shape::from((params.b_size, params.c_out, params.l_out())),
        "conv_transpose1d",
        |self_, layout, compute_dtype| {
            require_float(kernel.dtype(), "conv_transpose1d")?;
            if self_.dtype() != kernel.dtype() {
                return Err(Error::Msg(
                    "conv_transpose1d input and kernel dtype mismatch".into(),
                ));
            }

            let l_out = params.l_out();
            let dst_el = params.c_out * l_out * params.b_size;
            let out_shape = Shape::from((params.b_size, params.c_out, l_out));
            let device = self_.device();
            let out = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
            let out_layout = Layout::contiguous(&out_shape);

            let (src, src_layout) = ensure_contiguous(self_, layout)?;
            let (kernel_storage, kernel_layout) = ensure_contiguous(kernel, kernel_l)?;

            let uniforms = ConvTranspose1dUniforms {
                l_out: l_out as u32,
                stride: params.stride as u32,
                padding: params.padding as u32,
                output_padding: params.output_padding as u32,
                dilation: params.dilation as u32,
                dst_numel: dst_el as u32,
                _align: [0; 2],
                src_layout: TensorLayoutUniform::from_layout(&src_layout),
                kernel_layout: TensorLayoutUniform::from_layout(&kernel_layout),
                _pad: [0; 24],
            };

            let entry = format!("conv_transpose1d_{}", float_type_suffix(compute_dtype));
            let wgpu_kernel = WgpuKernel::compile_extended(
                device,
                conv_transpose1d_shader(compute_dtype),
                &entry,
                32,
            )
            .map_err(Error::from)?;
            let k_off = buffer_offset(&kernel_storage, &kernel_layout);
            let bind_group = wgpu_kernel
                .create_extended_bind_group(
                    device,
                    buffer_offset(&out, &out_layout),
                    buffer_offset(&src, &src_layout),
                    k_off.clone(),
                    k_off,
                    uniforms.as_bytes(),
                )
                .map_err(Error::from)?;
            let grid = workgroup_count(device, dst_el);
            self_
                .backing()
                .with_unmapped(|| {
                    wgpu_kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1])
                })
                .map_err(Error::from)?;
            Ok(out)
        },
    )
}

fn dispatch_pool2d(
    storage: &WgpuStorage,
    layout: &Layout,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    entry_base: &'static str,
    dtype: DType,
) -> Result<WgpuStorage> {
    let (k_h, k_w) = kernel_size;
    let (s_h, s_w) = stride;
    let (b, c, h, w) = layout
        .shape()
        .dims4()
        .map_err(|e| WgpuError::Message(format!("pool2d expects 4D input: {e}")))?;
    let h_out = (h - k_h) / s_h + 1;
    let w_out = (w - k_w) / s_w + 1;
    let dst_numel = b * c * h_out * w_out;
    let out_shape = Shape::from((b, c, h_out, w_out));
    let device = storage.device();
    let out = WgpuStorage::alloc(device, &out_shape, dtype)?;
    let out_layout = Layout::contiguous(&out_shape);
    let uniforms = Pool2dUniforms {
        k_h: k_h as u32,
        k_w: k_w as u32,
        s_h: s_h as u32,
        s_w: s_w as u32,
        dst_numel: dst_numel as u32,
        _align: [0; 3],
        src_layout: TensorLayoutUniform::from_layout(layout),
        _pad: [0; 44],
    };
    let entry = format!("{}_{}", entry_base, float_type_suffix(dtype));
    let kernel = compile_standard_kernel(device, pool2d_shader(dtype), &entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&out, &out_layout),
        buffer_offset(storage, layout),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = workgroup_count(device, dst_numel);
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(out)
}

fn pool_out_shape(
    layout: &Layout,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> CandleResult<Shape> {
    let (k_h, k_w) = kernel_size;
    let (s_h, s_w) = stride;
    let (b, c, h, w) = layout.shape().dims4()?;
    let h_out = (h - k_h) / s_h + 1;
    let w_out = (w - k_w) / s_w + 1;
    Ok(Shape::from((b, c, h_out, w_out)))
}

pub fn avg_pool2d(
    storage: &WgpuStorage,
    layout: &Layout,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> CandleResult<WgpuStorage> {
    let out_shape = pool_out_shape(layout, kernel_size, stride)?;
    with_compute_dtype(
        storage,
        layout,
        &out_shape,
        "avg_pool2d",
        |storage, layout, compute_dtype| {
            dispatch_pool2d(
                storage,
                layout,
                kernel_size,
                stride,
                "avg_pool2d",
                compute_dtype,
            )
            .map_err(Error::from)
        },
    )
}

pub fn max_pool2d(
    storage: &WgpuStorage,
    layout: &Layout,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> CandleResult<WgpuStorage> {
    let out_shape = pool_out_shape(layout, kernel_size, stride)?;
    with_compute_dtype(
        storage,
        layout,
        &out_shape,
        "max_pool2d",
        |storage, layout, compute_dtype| {
            dispatch_pool2d(
                storage,
                layout,
                kernel_size,
                stride,
                "max_pool2d",
                compute_dtype,
            )
            .map_err(Error::from)
        },
    )
}

pub fn upsample_nearest1d(
    storage: &WgpuStorage,
    layout: &Layout,
    dst_sz: usize,
) -> CandleResult<WgpuStorage> {
    let (b, c, _) = layout.shape().dims3()?;
    let out_shape = Shape::from((b, c, dst_sz));
    with_compute_dtype(
        storage,
        layout,
        &out_shape,
        "upsample_nearest1d",
        |storage, layout, compute_dtype| {
            let (b, c, src_sz) = layout.shape().dims3()?;
            let dst_numel = b * c * dst_sz;
            let out_shape = Shape::from((b, c, dst_sz));
            let device = storage.device();
            let out = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
            let out_layout = Layout::contiguous(&out_shape);
            let scale = src_sz as f32 / dst_sz as f32;
            let uniforms = UpsampleNearest1dUniforms {
                dst_sz: dst_sz as u32,
                scale_bits: scale.to_bits(),
                dst_numel: dst_numel as u32,
                _align: 0,
                src_layout: TensorLayoutUniform::from_layout(layout),
                _pad: [0; 48],
            };
            let entry = format!("upsample_nearest1d_{}", float_type_suffix(compute_dtype));
            let kernel =
                compile_standard_kernel(device, upsample_nearest1d_shader(compute_dtype), &entry)?;
            let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
                device.device(),
                device.queue(),
                buffer_offset(&out, &out_layout),
                buffer_offset(storage, layout),
                None,
                uniforms.as_bytes(),
            )?;
            let grid = workgroup_count(device, dst_numel);
            storage
                .backing()
                .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
                .map_err(Error::from)?;
            Ok(out)
        },
    )
}

pub fn upsample_nearest2d(
    storage: &WgpuStorage,
    layout: &Layout,
    dst_h: usize,
    dst_w: usize,
) -> CandleResult<WgpuStorage> {
    let (b, c, _, _) = layout.shape().dims4()?;
    let out_shape = Shape::from((b, c, dst_h, dst_w));
    with_compute_dtype(
        storage,
        layout,
        &out_shape,
        "upsample_nearest2d",
        |storage, layout, compute_dtype| {
            let (b, c, src_h, src_w) = layout.shape().dims4()?;
            let dst_numel = b * c * dst_h * dst_w;
            let out_shape = Shape::from((b, c, dst_h, dst_w));
            let device = storage.device();
            let out = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
            let out_layout = Layout::contiguous(&out_shape);
            let uniforms = UpsampleNearest2dUniforms {
                dst_h: dst_h as u32,
                dst_w: dst_w as u32,
                scale_h_bits: (src_h as f32 / dst_h as f32).to_bits(),
                scale_w_bits: (src_w as f32 / dst_w as f32).to_bits(),
                dst_numel: dst_numel as u32,
                _align: [0; 3],
                src_layout: TensorLayoutUniform::from_layout(layout),
                _pad: [0; 44],
            };
            let entry = format!("upsample_nearest2d_{}", float_type_suffix(compute_dtype));
            let kernel =
                compile_standard_kernel(device, upsample_nearest2d_shader(compute_dtype), &entry)?;
            let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
                device.device(),
                device.queue(),
                buffer_offset(&out, &out_layout),
                buffer_offset(storage, layout),
                None,
                uniforms.as_bytes(),
            )?;
            let grid = workgroup_count(device, dst_numel);
            storage
                .backing()
                .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
                .map_err(Error::from)?;
            Ok(out)
        },
    )
}

pub fn upsample_bilinear2d(
    storage: &WgpuStorage,
    layout: &Layout,
    dst_h: usize,
    dst_w: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
) -> CandleResult<WgpuStorage> {
    let (b, c, _, _) = layout.shape().dims4()?;
    let out_shape = Shape::from((b, c, dst_h, dst_w));
    with_compute_dtype(
        storage,
        layout,
        &out_shape,
        "upsample_bilinear2d",
        |storage, layout, compute_dtype| {
            let (b, c, _, _) = layout.shape().dims4()?;
            let dst_numel = b * c * dst_h * dst_w;
            let out_shape = Shape::from((b, c, dst_h, dst_w));
            let device = storage.device();
            let out = WgpuStorage::alloc(device, &out_shape, compute_dtype)?;
            let out_layout = Layout::contiguous(&out_shape);
            let uniforms = UpsampleBilinear2dUniforms {
                dst_h: dst_h as u32,
                dst_w: dst_w as u32,
                align_corners: u32::from(align_corners),
                has_scale_h: u32::from(scale_h.is_some()),
                scale_h_bits: scale_h.map(|s| s as f32).unwrap_or(0.0).to_bits(),
                has_scale_w: u32::from(scale_w.is_some()),
                scale_w_bits: scale_w.map(|s| s as f32).unwrap_or(0.0).to_bits(),
                dst_numel: dst_numel as u32,
                src_layout: TensorLayoutUniform::from_layout(layout),
                _pad: [0; 44],
            };
            let entry = format!("upsample_bilinear2d_{}", float_type_suffix(compute_dtype));
            let kernel =
                compile_standard_kernel(device, upsample_bilinear2d_shader(compute_dtype), &entry)?;
            let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
                device.device(),
                device.queue(),
                buffer_offset(&out, &out_layout),
                buffer_offset(storage, layout),
                None,
                uniforms.as_bytes(),
            )?;
            let grid = workgroup_count(device, dst_numel);
            storage
                .backing()
                .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
                .map_err(Error::from)?;
            Ok(out)
        },
    )
}
