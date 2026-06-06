use super::bind_group::{
    BindGroupBuilder, Copy2dUniforms, DequantUniforms, KernelUniforms, MatMulUniforms,
    QMatMulUniforms, QuantUniforms, ReduceUniforms, RmsNormUniforms, RopeUniforms, SdpaUniforms,
    SoftmaxUniforms, WhereUniforms,
};
use super::error::{Result, WgpuError};
use super::intel_caps::{tune_matmul_shader_source, IntelGeneration};
use super::kernel::WgpuKernel;
use super::storage::{buffer_offset, BufferBacking, BufferOffset, WgpuStorage, STORAGE_BUFFER_USAGE};
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::wgsl::{
    BINARY, BINARY_BF16, BINARY_F16, COPY, COPY2D, DEQUANT_Q4_0, DEQUANT_Q4_K, DEQUANT_Q5_0,
    DEQUANT_Q8_0, MATMUL_NAIVE, MATMUL_TILED, MATMUL_TILED_BF16, MATMUL_TILED_F16, MATMUL_TILED_VEC,
    MATMUL_VEC_WIDTH, QMATMUL_Q4_0, QMATMUL_Q4_K, QMATMUL_Q5_0, QMATMUL_Q8_0, QUANT_Q4_0,
    QUANT_Q5_0, QUANT_Q8_0, REDUCE, RMS_NORM, ROPE, SDPA_FULL, SDPA_VECTOR, SOFTMAX, UNARY,
    UNARY_BF16, UNARY_F16, WHERE_COND,
};
use crate::{CpuStorage, DType, Error, Layout, Result as CandleResult, Shape};
use std::sync::Arc;
use wgpu::BufferUsages;

fn elemwise_workgroup_count(device: &WgpuDevice, elem_count: usize) -> u32 {
    let wg = device.caps().elem_workgroup_size;
    (elem_count as u32).div_ceil(wg)
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
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
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
    TiledBf16,
    TiledVecBf16,
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
        DType::BF16 if caps.supports_native_bf16() => {
            if tiled && vec_aligned && large_enough {
                MatMulKernel::TiledVecBf16
            } else {
                MatMulKernel::TiledBf16
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
        MatMulKernel::TiledBf16 | MatMulKernel::TiledVecBf16 => MATMUL_TILED_BF16,
    }
}

fn matmul_entry_point(kernel: MatMulKernel) -> &'static str {
    match kernel {
        MatMulKernel::NaiveF32 => "matmul_naive_f32",
        MatMulKernel::TiledF32 => "matmul_tiled_f32",
        MatMulKernel::TiledVecF32 => "matmul_tiled_vec_f32",
        MatMulKernel::TiledF16 => "matmul_tiled_f16",
        MatMulKernel::TiledVecF16 => "matmul_tiled_vec_f16",
        MatMulKernel::TiledBf16 => "matmul_tiled_bf16",
        MatMulKernel::TiledVecBf16 => "matmul_tiled_vec_bf16",
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
        "usigmoid" => Some("sigmoid_f32"),
        "ugelu" => Some("gelu_f32"),
        "ugelu_erf" => Some("gelu_erf_f32"),
        "uabs" => Some("abs_f32"),
        "usin" => Some("sin_f32"),
        "ucos" => Some("cos_f32"),
        "utanh" => Some("tanh_f32"),
        "usqr" => Some("sqr_f32"),
        "uerf" => Some("erf_f32"),
        "uceil" => Some("ceil_f32"),
        "ufloor" => Some("floor_f32"),
        "uround" => Some("round_f32"),
        "usign" => Some("sign_f32"),
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

fn unary_entry_point_bf16(kernel: &str) -> Option<&'static str> {
    match unary_entry_point(kernel)? {
        "exp_f32" => Some("exp_bf16"),
        "neg_f32" => Some("neg_bf16"),
        "log_f32" => Some("log_bf16"),
        "sqrt_f32" => Some("sqrt_bf16"),
        "relu_f32" => Some("relu_bf16"),
        "recip_f32" => Some("recip_bf16"),
        "silu_f32" => Some("silu_bf16"),
        "sigmoid_f32" => Some("sigmoid_bf16"),
        "gelu_f32" => Some("gelu_bf16"),
        "gelu_erf_f32" => Some("gelu_erf_bf16"),
        "abs_f32" => Some("abs_bf16"),
        "sin_f32" => Some("sin_bf16"),
        "cos_f32" => Some("cos_bf16"),
        "tanh_f32" => Some("tanh_bf16"),
        "sqr_f32" => Some("sqr_bf16"),
        "erf_f32" => Some("erf_bf16"),
        "ceil_f32" => Some("ceil_bf16"),
        "floor_f32" => Some("floor_bf16"),
        "round_f32" => Some("round_bf16"),
        "sign_f32" => Some("sign_bf16"),
        _ => None,
    }
}

fn binary_entry_point_bf16(kernel: &str) -> Option<&'static str> {
    match binary_entry_point(kernel)? {
        "add_f32" => Some("add_bf16"),
        "sub_f32" => Some("sub_bf16"),
        "mul_f32" => Some("mul_bf16"),
        "div_f32" => Some("div_bf16"),
        "min_f32" => Some("min_bf16"),
        "max_f32" => Some("max_bf16"),
        _ => None,
    }
}

fn unary_entry_point_f16(kernel: &str) -> Option<&'static str> {
    match unary_entry_point(kernel)? {
        "exp_f32" => Some("exp_f16"),
        "neg_f32" => Some("neg_f16"),
        "log_f32" => Some("log_f16"),
        "sqrt_f32" => Some("sqrt_f16"),
        "relu_f32" => Some("relu_f16"),
        "recip_f32" => Some("recip_f16"),
        "silu_f32" => Some("silu_f16"),
        "sigmoid_f32" => Some("sigmoid_f16"),
        "gelu_f32" => Some("gelu_f16"),
        "gelu_erf_f32" => Some("gelu_erf_f16"),
        "abs_f32" => Some("abs_f16"),
        "sin_f32" => Some("sin_f16"),
        "cos_f32" => Some("cos_f16"),
        "tanh_f32" => Some("tanh_f16"),
        "sqr_f32" => Some("sqr_f16"),
        "erf_f32" => Some("erf_f16"),
        "ceil_f32" => Some("ceil_f16"),
        "floor_f32" => Some("floor_f16"),
        "round_f32" => Some("round_f16"),
        "sign_f32" => Some("sign_f16"),
        _ => None,
    }
}

fn binary_entry_point_f16(kernel: &str) -> Option<&'static str> {
    match binary_entry_point(kernel)? {
        "add_f32" => Some("add_f16"),
        "sub_f32" => Some("sub_f16"),
        "mul_f32" => Some("mul_f16"),
        "div_f32" => Some("div_f16"),
        "min_f32" => Some("min_f16"),
        "max_f32" => Some("max_f16"),
        _ => None,
    }
}

fn reduce_entry_point(op: ReduceOp) -> Option<&'static str> {
    match op {
        ReduceOp::Sum => Some("reduce_sum_f32"),
        ReduceOp::Max => Some("reduce_max_f32"),
        ReduceOp::Min => Some("reduce_min_f32"),
        ReduceOp::ArgMax => Some("reduce_argmax_f32"),
        ReduceOp::ArgMin => Some("reduce_argmin_f32"),
    }
}

fn compile_elemwise_kernel(
    device: &WgpuDevice,
    source: &str,
    entry_point: &str,
) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(source, device.caps());
    WgpuKernel::compile_with_workgroup_size(
        device,
        &tuned,
        entry_point,
        device.caps().elem_workgroup_size,
    )
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

struct ElemwiseBf16Dispatch<'a> {
    device: &'a WgpuDevice,
    source: &'a str,
    entry_point: &'a str,
    out: &'a WgpuStorage,
    out_layout: &'a Layout,
    in0: &'a WgpuStorage,
    in0_layout: &'a Layout,
    in1: Option<(&'a WgpuStorage, &'a Layout)>,
    backing: &'a super::storage::BufferBacking,
    uniforms: KernelUniforms,
}

/// Serial packed-bf16 elemwise dispatch (`@workgroup_size(1)` — safe RMW, not parallel).
fn dispatch_elemwise_bf16(args: ElemwiseBf16Dispatch<'_>) -> Result<()> {
    let ElemwiseBf16Dispatch {
        device,
        source,
        entry_point,
        out,
        out_layout,
        in0,
        in0_layout,
        in1,
        backing,
        uniforms,
    } = args;
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, source, entry_point, 1)?;
    let bind_group_builder = BindGroupBuilder::new();
    let in0_offset = buffer_offset(in0, in0_layout);
    let in1_offset = in1.map(|(storage, layout)| buffer_offset(storage, layout));
    let bind_group = bind_group_builder.create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(out, out_layout),
        in0_offset,
        in1_offset,
        uniforms.as_bytes(),
    )?;
    backing.with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [1, 1, 1]))
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
    let grid = elemwise_workgroup_count(device, out_layout.shape().elem_count());
    backing.with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
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
    let dtype = storage.dtype();
    let device = storage.device().clone();
    if dtype == DType::BF16 {
        if device.caps().supports_native_bf16() {
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new_affine(
                out_layout.shape().elem_count(),
                &out_layout,
                layout,
                mul,
                add,
            );
            dispatch_elemwise_bf16(ElemwiseBf16Dispatch {
                device: &device,
                source: UNARY_BF16,
                entry_point: "affine_bf16",
                out: &out,
                out_layout: &out_layout,
                in0: storage,
                in0_layout: layout,
                in1: None,
                backing: storage.backing(),
                uniforms,
            })
            .map_err(Error::from)?;
            return Ok(out);
        }
        let f32 = storage.to_dtype(layout, DType::F32)?;
        let f32_layout = Layout::contiguous(layout.shape());
        let out_f32 = dispatch_affine(&f32, &f32_layout, mul, add)?;
        return out_f32.to_dtype(layout, DType::BF16);
    }

    require_f32(dtype, "affine")?;

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
    let grid = elemwise_workgroup_count(&device, out_layout.shape().elem_count());
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(&device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_unary<B: UnaryOpT>(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<WgpuStorage> {
    let dtype = storage.dtype();
    let device = storage.device().clone();
    if dtype == DType::BF16 {
        if device.caps().supports_native_bf16() {
            let entry = unary_entry_point_bf16(B::KERNEL).ok_or_else(|| {
                WgpuError::Message(format!(
                    "wgpu bf16 unary kernel not available for op {}",
                    B::NAME
                ))
            })?;
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new(
                out_layout.shape().elem_count(),
                &out_layout,
                layout,
                None,
            );
            dispatch_elemwise_bf16(ElemwiseBf16Dispatch {
                device: &device,
                source: UNARY_BF16,
                entry_point: entry,
                out: &out,
                out_layout: &out_layout,
                in0: storage,
                in0_layout: layout,
                in1: None,
                backing: storage.backing(),
                uniforms,
            })
            .map_err(Error::from)?;
            return Ok(out);
        }
        let f32 = storage.to_dtype(layout, DType::F32)?;
        let f32_layout = Layout::contiguous(layout.shape());
        let out_f32 = dispatch_unary::<B>(&f32, &f32_layout)?;
        return out_f32.to_dtype(layout, DType::BF16);
    }

    if dtype == DType::F16 {
        if device.caps().supports_native_f16() {
            let entry = unary_entry_point_f16(B::KERNEL).ok_or_else(|| {
                WgpuError::Message(format!(
                    "wgpu f16 unary kernel not available for op {}",
                    B::NAME
                ))
            })?;
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::F16)?;
            let out_layout = Layout::contiguous(out_shape);
            dispatch_elemwise(ElemwiseDispatch {
                device: &device,
                source: UNARY_F16,
                entry_point: entry,
                out: &out,
                out_layout: &out_layout,
                in0: storage,
                in0_layout: layout,
                in1: None,
                backing: storage.backing(),
            })
            .map_err(Error::from)?;
            return Ok(out);
        }
        let f32 = storage.to_dtype(layout, DType::F32)?;
        let f32_layout = Layout::contiguous(layout.shape());
        let out_f32 = dispatch_unary::<B>(&f32, &f32_layout)?;
        return out_f32.to_dtype(layout, DType::F16);
    }

    require_f32(dtype, B::NAME)?;
    let entry = unary_entry_point(B::KERNEL).ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu unary kernel not available for op {}",
            B::NAME
        ))
    })?;

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
    let dtype = lhs.dtype();
    if dtype != rhs.dtype() {
        return Err(Error::UnsupportedDTypeForOp(dtype, B::NAME).bt());
    }
    let device = lhs.device().clone();
    if dtype == DType::BF16 {
        if device.caps().supports_native_bf16() {
            let entry = binary_entry_point_bf16(B::KERNEL).ok_or_else(|| {
                WgpuError::Message(format!(
                    "wgpu bf16 binary kernel not available for op {}",
                    B::NAME
                ))
            })?;
            let out_shape = lhs_layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new(
                out_layout.shape().elem_count(),
                &out_layout,
                lhs_layout,
                Some(rhs_layout),
            );
            dispatch_elemwise_bf16(ElemwiseBf16Dispatch {
                device: &device,
                source: BINARY_BF16,
                entry_point: entry,
                out: &out,
                out_layout: &out_layout,
                in0: lhs,
                in0_layout: lhs_layout,
                in1: Some((rhs, rhs_layout)),
                backing: lhs.backing(),
                uniforms,
            })
            .map_err(Error::from)?;
            return Ok(out);
        }
        let lhs_f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
        let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
        let lhs_layout = Layout::contiguous(lhs_layout.shape());
        let rhs_layout = Layout::contiguous(rhs_layout.shape());
        let out_f32 = dispatch_binary::<B>(&lhs_f32, &rhs_f32, &lhs_layout, &rhs_layout)?;
        let out_layout = Layout::contiguous(lhs_layout.shape());
        return out_f32.to_dtype(&out_layout, DType::BF16);
    }

    if dtype == DType::F16 {
        if device.caps().supports_native_f16() {
            let entry = binary_entry_point_f16(B::KERNEL).ok_or_else(|| {
                WgpuError::Message(format!(
                    "wgpu f16 binary kernel not available for op {}",
                    B::NAME
                ))
            })?;
            let out_shape = lhs_layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::F16)?;
            let out_layout = Layout::contiguous(out_shape);
            dispatch_elemwise(ElemwiseDispatch {
                device: &device,
                source: BINARY_F16,
                entry_point: entry,
                out: &out,
                out_layout: &out_layout,
                in0: lhs,
                in0_layout: lhs_layout,
                in1: Some((rhs, rhs_layout)),
                backing: lhs.backing(),
            })
            .map_err(Error::from)?;
            return Ok(out);
        }
        let lhs_f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
        let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
        let lhs_layout = Layout::contiguous(lhs_layout.shape());
        let rhs_layout = Layout::contiguous(rhs_layout.shape());
        let out_f32 = dispatch_binary::<B>(&lhs_f32, &rhs_f32, &lhs_layout, &rhs_layout)?;
        let out_layout = Layout::contiguous(lhs_layout.shape());
        return out_f32.to_dtype(&out_layout, DType::F16);
    }

    require_f32(dtype, B::NAME)?;
    let entry = binary_entry_point(B::KERNEL).ok_or_else(|| {
        WgpuError::Message(format!(
            "wgpu binary kernel not available for op {}",
            B::NAME
        ))
    })?;

    let out_shape = lhs_layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::F32)?;
    let out_layout = Layout::contiguous(out_shape);
    dispatch_elemwise(ElemwiseDispatch {
        device: &device,
        source: BINARY,
        entry_point: entry,
        out: &out,
        out_layout: &out_layout,
        in0: lhs,
        in0_layout: lhs_layout,
        in1: Some((rhs, rhs_layout)),
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

    if storage_dtype == DType::BF16 && !device.caps().supports_native_bf16() {
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
        return out_f32.to_dtype(&out_layout, DType::BF16);
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

fn compile_qmatmul_kernel(
    device: &WgpuDevice,
    source: &str,
    entry_point: &str,
) -> Result<WgpuKernel> {
    WgpuKernel::compile_with_workgroup_size(device, source, entry_point, 8)
}

fn dispatch_qmatmul(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    source: &str,
    entry_point: &str,
    k_block: usize,
) -> CandleResult<WgpuStorage> {
    require_f32(lhs.dtype(), "qmatmul")?;
    if !k.is_multiple_of(k_block) {
        return Err(WgpuError::Message(format!(
            "qmatmul k={k} must be divisible by {k_block}"
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
    let kernel = compile_qmatmul_kernel(&device, source, entry_point).map_err(Error::from)?;
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

/// Quantized Q4_0 matrix multiply: `dst = lhs @ rhs^T` with f32 activations.
pub fn dispatch_qmatmul_q4_0(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    dims: (usize, usize, usize, usize),
    lhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_qmatmul(
        lhs,
        rhs_buffer,
        dims,
        lhs_layout,
        QMATMUL_Q4_0,
        "qmatmul_q4_0_f32",
        32,
    )
}

/// Quantized Q5_0 matrix multiply: `dst = lhs @ rhs^T` with f32 activations.
pub fn dispatch_qmatmul_q5_0(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    dims: (usize, usize, usize, usize),
    lhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_qmatmul(
        lhs,
        rhs_buffer,
        dims,
        lhs_layout,
        QMATMUL_Q5_0,
        "qmatmul_q5_0_f32",
        32,
    )
}

/// Quantized Q8_0 matrix multiply: `dst = lhs @ rhs^T` with f32 activations.
pub fn dispatch_qmatmul_q8_0(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    dims: (usize, usize, usize, usize),
    lhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_qmatmul(
        lhs,
        rhs_buffer,
        dims,
        lhs_layout,
        QMATMUL_Q8_0,
        "qmatmul_q8_0_f32",
        32,
    )
}

/// Quantized Q4_K matrix multiply: `dst = lhs @ rhs^T` with f32 activations.
pub fn dispatch_qmatmul_q4_k(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    dims: (usize, usize, usize, usize),
    lhs_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_qmatmul(
        lhs,
        rhs_buffer,
        dims,
        lhs_layout,
        QMATMUL_Q4_K,
        "qmatmul_q4_k_f32",
        256,
    )
}

/// Upload quantized weight bytes to a GPU storage buffer.
pub fn upload_quant_weights(device: &WgpuDevice, bytes: &[u8]) -> Result<Arc<wgpu::Buffer>> {
    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("candle quant weights"),
        size: bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device.queue().write_buffer(&buffer, 0, bytes);
    Ok(Arc::new(buffer))
}

/// Upload Q4_0 weight bytes to a GPU storage buffer.
pub fn upload_q4_0_weights(device: &WgpuDevice, bytes: &[u8]) -> Result<Arc<wgpu::Buffer>> {
    upload_quant_weights(device, bytes)
}

/// Whether [`dispatch_dequant_f32`] supports this GGML dtype on GPU.
pub fn gpu_dequant_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::Q4K
    )
}

/// Whether [`dispatch_quant_f32`] supports this GGML dtype on GPU.
pub fn gpu_quant_supported(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0)
}

fn dispatch_dequant_unary(
    device: &WgpuDevice,
    source: &str,
    entry_point: &str,
    workgroup_size: u32,
    grid_x: u32,
    quant_buffer: &wgpu::Buffer,
    elem_count: usize,
) -> CandleResult<WgpuStorage> {
    let out_shape = Shape::from(elem_count);
    let out = WgpuStorage::alloc(device, &out_shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(&out_shape);
    let uniforms = DequantUniforms {
        elem_count: elem_count as u32,
        _pad: [0; 71],
    };
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, source, entry_point, workgroup_size)
            .map_err(Error::from)?;
    let bind_group_builder = BindGroupBuilder::new();
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            BufferOffset {
                buffer: quant_buffer,
                offset_in_bytes: 0,
            },
            None,
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    out.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid_x, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

/// Dequantize a GGML buffer on GPU into f32 storage.
pub fn dispatch_dequant_f32(
    device: &WgpuDevice,
    dtype: GgmlDType,
    quant_buffer: &wgpu::Buffer,
    elem_count: usize,
) -> CandleResult<WgpuStorage> {
    if elem_count == 0 {
        return WgpuStorage::alloc(device, &Shape::from(0usize), DType::F32).map_err(Error::from);
    }
    match dtype {
        GgmlDType::Q4_0 => dispatch_dequant_unary(
            device,
            DEQUANT_Q4_0,
            "dequant_q4_0_f32",
            256,
            elem_count.div_ceil(256) as u32,
            quant_buffer,
            elem_count,
        ),
        GgmlDType::Q5_0 => dispatch_dequant_unary(
            device,
            DEQUANT_Q5_0,
            "dequant_q5_0_f32",
            256,
            elem_count.div_ceil(256) as u32,
            quant_buffer,
            elem_count,
        ),
        GgmlDType::Q8_0 => dispatch_dequant_unary(
            device,
            DEQUANT_Q8_0,
            "dequant_q8_0_f32",
            256,
            elem_count.div_ceil(256) as u32,
            quant_buffer,
            elem_count,
        ),
        GgmlDType::Q4K => {
            let block_size = GgmlDType::Q4K.block_size();
            if !elem_count.is_multiple_of(block_size) {
                return Err(Error::Msg(format!(
                    "q4k dequant requires elem_count multiple of {block_size}"
                )));
            }
            dispatch_dequant_unary(
                device,
                DEQUANT_Q4_K,
                "dequant_q4_k_f32",
                32,
                (elem_count / block_size) as u32,
                quant_buffer,
                elem_count,
            )
        }
        other => Err(Error::Msg(format!(
            "wgpu gpu dequant unsupported for {other:?}"
        ))),
    }
}

/// Quantize contiguous f32 activations on GPU into a GGML buffer.
pub fn dispatch_quant_f32(
    device: &WgpuDevice,
    dtype: GgmlDType,
    src: &WgpuStorage,
    src_layout: &Layout,
) -> CandleResult<Arc<wgpu::Buffer>> {
    require_f32(src.dtype(), "quantize")?;
    if !src_layout.is_contiguous() {
        return Err(Error::Msg("wgpu quantize requires contiguous f32 input".into()));
    }
    let elem_count = src_layout.shape().elem_count();
    let block_size = dtype.block_size();
    if !elem_count.is_multiple_of(block_size) {
        return Err(Error::Msg(format!(
            "wgpu quantize requires elem_count multiple of {block_size}"
        )));
    }
    let num_blocks = elem_count / block_size;
    let out_bytes = num_blocks * dtype.type_size();
    let (source, entry) = match dtype {
        GgmlDType::Q4_0 => (QUANT_Q4_0, "quant_q4_0_f32"),
        GgmlDType::Q5_0 => (QUANT_Q5_0, "quant_q5_0_f32"),
        GgmlDType::Q8_0 => (QUANT_Q8_0, "quant_q8_0_f32"),
        other => return Err(Error::Msg(format!("wgpu gpu quant unsupported for {other:?}"))),
    };

    let buffer = Arc::new(device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("candle quant output"),
        size: out_bytes as u64,
        usage: STORAGE_BUFFER_USAGE,
        mapped_at_creation: false,
    }));
    device
        .queue()
        .write_buffer(&buffer, 0, &vec![0u8; out_bytes]);

    let out_layout = Layout::contiguous(Shape::from(out_bytes));
    let out_storage = WgpuStorage::new(
        BufferBacking::DeviceLocal(buffer.clone()),
        device.clone(),
        out_bytes,
        DType::U8,
    );
    let uniforms = QuantUniforms {
        elem_count: elem_count as u32,
        _pad: [0; 71],
    };
    let kernel = WgpuKernel::compile_with_workgroup_size(device, source, entry, 1)
        .map_err(Error::from)?;
    let bind_group_builder = BindGroupBuilder::new();
    let bind_group = bind_group_builder
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out_storage, &out_layout),
            buffer_offset(src, src_layout),
            None,
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    src.backing()
        .with_unmapped(|| {
            kernel.dispatch_bind_group(device, &bind_group, [num_blocks as u32, 1, 1])
        })
        .map_err(Error::from)?;
    Ok(buffer)
}

pub fn dispatch_reduce(
    storage: &WgpuStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), "reduce")?;

    if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) && reduce_dims.len() != 1 {
        return Err(Error::OnlySingleDimension {
            op: if matches!(op, ReduceOp::ArgMax) {
                "argmax"
            } else {
                "argmin"
            },
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
    let arg_index = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
    if dst_elem_count == 0 {
        let dtype = if arg_index { DType::U32 } else { DType::F32 };
        return WgpuStorage::alloc(storage.device(), &dst_shape, dtype).map_err(Error::from);
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
    if arg_index {
        return arg_index_f32_to_u32(out);
    }
    Ok(out)
}

/// Argmax/argmin kernels write indices as f32; Candle expects `U32` storage.
fn arg_index_f32_to_u32(out_f32: WgpuStorage) -> CandleResult<WgpuStorage> {
    let cpu = out_f32.to_cpu_storage()?;
    let CpuStorage::F32(indices) = cpu else {
        return Err(Error::Msg("expected f32 arg index buffer".into()));
    };
    let indices: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
    WgpuStorage::from_cpu(out_f32.device(), &CpuStorage::U32(indices)).map_err(Error::from)
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

/// Parameters for a 2D strided copy between wgpu buffers.
pub struct Copy2dParams {
    pub d1: usize,
    pub d2: usize,
    pub src_stride: usize,
    pub dst_stride: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
}

pub fn dispatch_copy2d(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    params: Copy2dParams,
) -> Result<()> {
    if src.dtype() != DType::F32 || dst.dtype() != DType::F32 {
        return Err(WgpuError::Message("wgpu copy2d only supports f32".into()));
    }
    let device = src.device();
    let uniforms = Copy2dUniforms {
        d1: params.d1 as u32,
        d2: params.d2 as u32,
        src_stride: params.src_stride as u32,
        dst_stride: params.dst_stride as u32,
        src_offset: params.src_offset as u32,
        dst_offset: params.dst_offset as u32,
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
    let total = (params.d1 * params.d2) as u32;
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

/// Fused softmax along the last dimension (contiguous f32 input).
pub fn dispatch_softmax_last_dim_f32(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<WgpuStorage> {
    require_f32(storage.dtype(), "softmax")?;
    if !layout.is_contiguous() {
        return Err(Error::Msg("softmax requires contiguous input".into()));
    }
    let dims = layout.dims();
    let last_dim = *dims
        .last()
        .ok_or_else(|| Error::Msg("empty tensor in softmax".into()))?;
    let n_rows = layout.shape().elem_count() / last_dim;
    let device = storage.device();
    let out = WgpuStorage::alloc(device, layout.shape(), DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(layout.shape());
    let uniforms = SoftmaxUniforms {
        n_rows: n_rows as u32,
        last_dim: last_dim as u32,
        _pad: [0; 70],
    };
    let tuned = super::intel_caps::tune_shader_source(SOFTMAX, device.caps());
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, &tuned, "softmax_last_dim_f32", 32)
            .map_err(Error::from)?;
    let in0 = buffer_offset(storage, layout);
    let bind_group = BindGroupBuilder::new()
        .create_bind_group_bytes(
            device.device(),
            device.queue(),
            buffer_offset(&out, &out_layout),
            in0.clone(),
            Some(in0),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [n_rows as u32, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

fn sdpa_dummy_mask_buffer(device: &WgpuDevice) -> Result<Arc<wgpu::Buffer>> {
    device.allocate_buffer(4, STORAGE_BUFFER_USAGE)
}

fn validate_sdpa_shapes(
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
) -> CandleResult<(usize, usize, usize, usize, usize, usize, usize)> {
    if !(q_layout.is_contiguous() && k_layout.is_contiguous() && v_layout.is_contiguous()) {
        return Err(Error::Msg("sdpa requires contiguous q, k, v".into()));
    }

    let (bs, n_q_heads, q_seq, head_dim) = q_layout.shape().dims4()?;
    let (k_bs, n_kv_heads, k_seq, k_head) = k_layout.shape().dims4()?;
    let (v_bs, v_kv_heads, v_k_seq, v_dim) = v_layout.shape().dims4()?;

    if bs != k_bs || bs != v_bs {
        return Err(Error::Msg("sdpa batch size mismatch".into()));
    }
    if n_kv_heads != v_kv_heads || k_seq != v_k_seq || head_dim != k_head {
        return Err(Error::Msg("sdpa k/v shape mismatch".into()));
    }
    if n_q_heads % n_kv_heads != 0 {
        return Err(Error::Msg("sdpa n_heads must be a multiple of n_kv_heads".into()));
    }
    if head_dim != 64 && head_dim != 128 {
        return Err(Error::Msg(format!(
            "sdpa supports head_dim 64 or 128, got {head_dim}"
        )));
    }
    if v_dim > 128 {
        return Err(Error::Msg(format!("sdpa supports v_dim <= 128, got {v_dim}")));
    }

    Ok((bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim))
}

/// Fused scaled dot-product attention with vector/full routing (f32).
pub fn dispatch_sdpa_f32(
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
    mask: Option<(&WgpuStorage, &Layout)>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> CandleResult<WgpuStorage> {
    require_f32(q.dtype(), "sdpa")?;
    require_f32(k.dtype(), "sdpa")?;
    require_f32(v.dtype(), "sdpa")?;

    let (bs, _n_q_heads, _n_kv_heads, q_seq, k_seq, _head_dim, _v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;

    if q_seq > k_seq {
        return Err(Error::Msg("sdpa requires q_seq <= k_seq".into()));
    }

    let use_vector = q_seq <= 8 && mask.is_none() && !do_causal;
    if use_vector {
        return dispatch_sdpa_vector_f32(q, k, v, q_layout, k_layout, v_layout, scale, softcapping);
    }

    if softcapping != 1.0 {
        return Err(Error::Msg(
            "wgpu sdpa_full does not support softcapping (must be 1.0)".into(),
        ));
    }

    if let Some((mask_storage, mask_layout)) = mask {
        require_f32(mask_storage.dtype(), "sdpa mask")?;
        if !mask_layout.is_contiguous() {
            return Err(Error::Msg("sdpa mask must be contiguous".into()));
        }
        let mask_dims = mask_layout.shape().dims4()?;
        if mask_dims != (bs, _n_q_heads, q_seq, k_seq) {
            return Err(Error::Msg(format!(
                "sdpa mask shape must be ({bs}, {}, {q_seq}, {k_seq}), got {mask_dims:?}",
                _n_q_heads
            )));
        }
    }

    dispatch_sdpa_full_f32(
        q,
        k,
        v,
        q_layout,
        k_layout,
        v_layout,
        mask,
        do_causal,
        scale,
    )
}

/// Fused scaled dot-product attention (prefill/full path, f32).
pub fn dispatch_sdpa_full_f32(
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
    mask: Option<(&WgpuStorage, &Layout)>,
    do_causal: bool,
    scale: f32,
) -> CandleResult<WgpuStorage> {
    let (bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;

    let out_shape = Shape::from((bs, n_q_heads, q_seq, v_dim));
    let device = q.device();
    let out = WgpuStorage::alloc(device, &out_shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(&out_shape);
    let gqa_factor = n_q_heads / n_kv_heads;
    let has_mask = u32::from(mask.is_some());
    let uniforms = SdpaUniforms {
        bs: bs as u32,
        n_q_heads: n_q_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        q_seq: q_seq as u32,
        k_seq: k_seq as u32,
        head_dim: head_dim as u32,
        v_dim: v_dim as u32,
        gqa_factor: gqa_factor as u32,
        scale_bits: scale.to_bits(),
        softcapping_bits: 1.0f32.to_bits(),
        has_mask,
        do_causal: u32::from(do_causal),
        ql_off: (k_seq - q_seq) as u32,
        _pad: [0; 59],
    };

    let dummy_mask = if mask.is_some() {
        None
    } else {
        Some(sdpa_dummy_mask_buffer(device)?)
    };
    let mask_binding = if let Some((mask_storage, mask_layout)) = mask {
        buffer_offset(mask_storage, mask_layout)
    } else {
        let dummy = dummy_mask.as_ref().expect("dummy mask buffer");
        BufferOffset {
            buffer: Arc::as_ref(dummy),
            offset_in_bytes: 0,
        }
    };

    let kernel =
        WgpuKernel::compile_sdpa(device, SDPA_FULL, "sdpa_full_f32", 1).map_err(Error::from)?;
    let bind_group = kernel
        .create_sdpa_bind_group(
            device,
            buffer_offset(&out, &out_layout),
            buffer_offset(q, q_layout),
            buffer_offset(k, k_layout),
            buffer_offset(v, v_layout),
            mask_binding,
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    let grid = (bs * n_q_heads * q_seq) as u32;
    q.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

/// Fused scaled dot-product attention (vector/decode path, f32).
pub fn dispatch_sdpa_vector_f32(
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
    scale: f32,
    softcapping: f32,
) -> CandleResult<WgpuStorage> {
    let (bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;
    if q_seq > 8 {
        return Err(Error::Msg(format!(
            "sdpa_vector supports q_seq <= 8, got {q_seq}"
        )));
    }

    let out_shape = Shape::from((bs, n_q_heads, q_seq, v_dim));
    let device = q.device();
    let out = WgpuStorage::alloc(device, &out_shape, DType::F32).map_err(Error::from)?;
    let out_layout = Layout::contiguous(&out_shape);
    let gqa_factor = n_q_heads / n_kv_heads;
    let uniforms = SdpaUniforms {
        bs: bs as u32,
        n_q_heads: n_q_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        q_seq: q_seq as u32,
        k_seq: k_seq as u32,
        head_dim: head_dim as u32,
        v_dim: v_dim as u32,
        gqa_factor: gqa_factor as u32,
        scale_bits: scale.to_bits(),
        softcapping_bits: softcapping.to_bits(),
        has_mask: 0,
        do_causal: 0,
        ql_off: 0,
        _pad: [0; 59],
    };
    let kernel =
        WgpuKernel::compile_extended(device, SDPA_VECTOR, "sdpa_vector_f32", 1).map_err(Error::from)?;
    let bind_group = kernel
        .create_extended_bind_group(
            device,
            buffer_offset(&out, &out_layout),
            buffer_offset(q, q_layout),
            buffer_offset(k, k_layout),
            buffer_offset(v, v_layout),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    let grid = (bs * n_q_heads * q_seq) as u32;
    q.backing()
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
        assert_eq!(unary_entry_point("usin"), Some("sin_f32"));
        assert_eq!(unary_entry_point("ucos"), Some("cos_f32"));
        assert_eq!(unary_entry_point("utanh"), Some("tanh_f32"));
        assert_eq!(unary_entry_point("usqr"), Some("sqr_f32"));
        assert_eq!(unary_entry_point("uerf"), Some("erf_f32"));
        assert_eq!(unary_entry_point("ugelu_erf"), Some("gelu_erf_f32"));
        assert_eq!(unary_entry_point("usigmoid"), Some("sigmoid_f32"));
        assert_eq!(unary_entry_point("usign"), Some("sign_f32"));
        assert_eq!(unary_entry_point("unknown"), None);
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
        assert_eq!(reduce_entry_point(ReduceOp::ArgMax), Some("reduce_argmax_f32"));
        assert_eq!(reduce_entry_point(ReduceOp::ArgMin), Some("reduce_argmin_f32"));
    }

    #[test]
    fn broadcast_3d_stride_pattern() {
        let layout = Layout::contiguous(&Shape::from((1, 8, 1)))
            .broadcast_as(Shape::from((1, 8, 16)))
            .unwrap();
        assert_eq!(layout.stride(), &[8, 1, 0]);
    }

    #[test]
    fn binary_broadcast_dispatches_on_noop_device() {
        use crate::op::Add;

        let device = super::super::WgpuDevice::new_test(true, 1024);
        let lhs = WgpuStorage::from_cpu(
            &device,
            &CpuStorage::F32((0..8).map(|i| i as f32).collect()),
        )
        .unwrap();
        let lhs_layout = Layout::contiguous(&Shape::from((1, 8, 1)))
            .broadcast_as(Shape::from((1, 8, 16)))
            .unwrap();
        let rhs = WgpuStorage::from_cpu(&device, &CpuStorage::F32(vec![1.0; 128])).unwrap();
        let rhs_layout = Layout::contiguous(&Shape::from((1, 8, 16)));
        dispatch_binary::<Add>(&lhs, &rhs, &lhs_layout, &rhs_layout).unwrap();
    }

    #[test]
    fn elemwise_kernel_uses_adapter_workgroup_size() {
        let device = super::super::WgpuDevice::new_test(true, 1024);
        let kernel = compile_elemwise_kernel(&device, UNARY, "exp_f32").unwrap();
        let wg = device.caps().elem_workgroup_size;
        assert_eq!(kernel.workgroup_size(), wg);
        assert_eq!(elemwise_workgroup_count(&device, 100), 100u32.div_ceil(wg));
    }

    #[test]
    fn cast_bf16_roundtrip_noop_device() {
        let device = super::super::WgpuDevice::new_test(true, 1024);
        let storage = WgpuStorage::from_cpu(
            &device,
            &CpuStorage::F32(vec![1.0, -2.5, 0.0, 3.25]),
        )
        .unwrap();
        let layout = Layout::contiguous(&Shape::from((4,)));
        let bf16 = storage.to_dtype(&layout, DType::BF16).unwrap();
        let back = bf16.to_dtype(&layout, DType::F32).unwrap();
        let cpu = back.to_cpu_storage().unwrap();
        let CpuStorage::F32(v) = cpu else {
            panic!("expected f32");
        };
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn p1_shaders_compile_on_noop_device() {
        let device = super::super::WgpuDevice::new_test(true, 1024);
        compile_qmatmul_kernel(&device, QMATMUL_Q8_0, "qmatmul_q8_0_f32").unwrap();
        compile_qmatmul_kernel(&device, QMATMUL_Q4_K, "qmatmul_q4_k_f32").unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, SOFTMAX, "softmax_last_dim_f32", 32)
            .unwrap();
        WgpuKernel::compile_extended(&device, SDPA_VECTOR, "sdpa_vector_f32", 1).unwrap();
        WgpuKernel::compile_sdpa(&device, SDPA_FULL, "sdpa_full_f32", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, QMATMUL_Q5_0, "qmatmul_q5_0_f32", 8)
            .unwrap();
        compile_matmul_kernel(&device, MatMulKernel::TiledBf16).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, UNARY_BF16, "gelu_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, UNARY_BF16, "affine_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_BF16, "add_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_BF16, "min_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, DEQUANT_Q4_0, "dequant_q4_0_f32", 256)
            .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, DEQUANT_Q5_0, "dequant_q5_0_f32", 256)
            .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, DEQUANT_Q8_0, "dequant_q8_0_f32", 256)
            .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, DEQUANT_Q4_K, "dequant_q4_k_f32", 32)
            .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, QUANT_Q4_0, "quant_q4_0_f32", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, QUANT_Q5_0, "quant_q5_0_f32", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, QUANT_Q8_0, "quant_q8_0_f32", 1).unwrap();
    }
}
