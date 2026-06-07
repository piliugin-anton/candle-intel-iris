use super::bind_group::{
    BindGroupBuilder, Copy2dUniforms, DequantUniforms, KernelUniforms, MatMulUniforms,
    QMatMulUniforms, QuantUniforms, ReduceUniforms, RmsNormUniforms, RopeIUniforms,
    RopeThdUniforms, RopeUniforms, SdpaUniforms, SoftmaxUniforms, WhereUniforms,
};
use super::error::{Result, WgpuError};
use super::intel_caps::{tune_matmul_shader_source, IntelGeneration};
use super::kernel::{elemwise_workgroup_count, per_elem_dispatch_grid, workgroup_count, WgpuKernel};
use super::storage::{
    buffer_offset, BufferBacking, BufferOffset, WgpuStorage, STORAGE_BUFFER_USAGE,
};
use super::WgpuDevice;
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::wgsl::{
    BINARY, BINARY_BF16, BINARY_F16, BINARY_I32, BINARY_U32, CAST, CMP, CMP_BF16, CMP_F16, COPY,
    COPY2D, COPY2D_BF16,
    COPY2D_F16, COPY_BF16, COPY_F16, COPY_U32, DEQUANT_Q4_0, DEQUANT_Q4_K, DEQUANT_Q5_0,
    DEQUANT_Q8_0, LAYER_NORM, LAYER_NORM_BF16, LAYER_NORM_F16, MATMUL_GEMV, MATMUL_NAIVE,
    MATMUL_TILED,
    MATMUL_TILED_BF16, MATMUL_TILED_BF16ACC, MATMUL_TILED_F16, MATMUL_TILED_F16ACC,
    MATMUL_TILED_VEC, QMATMUL_Q4_0, QMATMUL_Q4_0_F16, QMATMUL_Q4_K,
    QMATMUL_Q4_K_F16, QMATMUL_Q5_0, QMATMUL_Q5_0_F16, QMATMUL_Q8_0, QMATMUL_Q8_0_F16,
    QUANT_Q4_0, QUANT_Q5_0, QUANT_Q8_0, REDUCE, REDUCE_BF16, REDUCE_F16, RMS_NORM, RMS_NORM_BF16,
    RMS_NORM_F16, ROPE, ROPE_BF16, ROPE_F16, ROPE_I, ROPE_I_BF16, ROPE_I_F16, ROPE_THD,
    ROPE_THD_BF16, ROPE_THD_F16, SDPA_FULL, SDPA_FULL_BF16, SDPA_FULL_F16, SDPA_VECTOR,
    SDPA_VECTOR_BF16, SDPA_VECTOR_F16, SDPA_WORKGROUP_SIZE, SOFTMAX, SOFTMAX_BF16, SOFTMAX_F16,
    UNARY, UNARY_BF16, UNARY_F16, WHERE_COND, WHERE_COND_BF16, WHERE_COND_F16,
};
use crate::{DType, Error, Layout, Result as CandleResult, Shape};
use std::sync::Arc;
use wgpu::BufferUsages;

fn require_f32(dtype: DType, op: &'static str) -> CandleResult<()> {
    if dtype == DType::F32 {
        Ok(())
    } else {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    }
}

pub(crate) fn require_float(dtype: DType, op: &'static str) -> CandleResult<()> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        other => Err(Error::UnsupportedDTypeForOp(other, op).bt()),
    }
}

pub(crate) fn float_type_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "f32",
    }
}

fn copy_shader_source(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => COPY_F16,
        DType::BF16 => COPY_BF16,
        _ => COPY,
    }
}

fn copy2d_shader_source(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => COPY2D_F16,
        DType::BF16 => COPY2D_BF16,
        _ => COPY2D,
    }
}

fn reduce_shader_source(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => REDUCE_F16,
        DType::BF16 => REDUCE_BF16,
        _ => REDUCE,
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
    GemvF32,
    GemvColF32,
    NaiveF32,
    TiledF32,
    TiledVecF32,
    TiledF16,
    TiledVecF16,
    TiledF16AccF32,
    TiledVecF16AccF32,
    TiledBf16,
    TiledVecBf16,
    TiledBf16AccF32,
    TiledVecBf16AccF32,
}

fn select_matmul_kernel(
    device: &WgpuDevice,
    storage_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
) -> MatMulKernel {
    let caps = device.caps();
    let compute_dtype = caps.effective_compute_dtype(storage_dtype);
    let tiled = caps.generation != IntelGeneration::NonIntel;
    let tile = caps.matmul_tile_size as usize;
    let vec_width = caps.matmul_vec_width as usize;
    let vec_aligned = k.is_multiple_of(vec_width) && n.is_multiple_of(vec_width);
    let large_enough = m >= 8 || n >= 8 || k >= 8;
    // Tiled kernels pay 2 barriers per K-tile; prefer naive when M or N is thinner than a tile.
    let use_tiled = tiled && large_enough && m >= tile && n >= tile;

    match (storage_dtype, compute_dtype) {
        (DType::F16, DType::F32) if caps.supports_shader_f16 => {
            if use_tiled && vec_aligned {
                MatMulKernel::TiledVecF16AccF32
            } else {
                MatMulKernel::TiledF16AccF32
            }
        }
        (DType::F16, DType::F16) if caps.supports_native_f16() => {
            if use_tiled && vec_aligned {
                MatMulKernel::TiledVecF16
            } else {
                MatMulKernel::TiledF16
            }
        }
        (DType::BF16, DType::F32) => {
            if use_tiled && vec_aligned {
                MatMulKernel::TiledVecBf16AccF32
            } else {
                MatMulKernel::TiledBf16AccF32
            }
        }
        (DType::BF16, DType::BF16) if caps.supports_native_bf16() => {
            if use_tiled && vec_aligned {
                MatMulKernel::TiledVecBf16
            } else {
                MatMulKernel::TiledBf16
            }
        }
        _ => {
            if use_tiled && vec_aligned {
                MatMulKernel::TiledVecF32
            } else if use_tiled {
                MatMulKernel::TiledF32
            } else if m == 1 {
                MatMulKernel::GemvF32
            } else if n == 1 {
                MatMulKernel::GemvColF32
            } else {
                MatMulKernel::NaiveF32
            }
        }
    }
}

fn matmul_dispatch_grid(
    kernel: MatMulKernel,
    device: &WgpuDevice,
    b: usize,
    m: usize,
    n: usize,
) -> [u32; 3] {
    match kernel {
        MatMulKernel::GemvF32 => {
            let wg = device.caps().elem_workgroup_size;
            [workgroup_count(wg, n), 1, b as u32]
        }
        MatMulKernel::GemvColF32 => {
            let wg = device.caps().elem_workgroup_size;
            [workgroup_count(wg, m), 1, b as u32]
        }
        _ => {
            let wg = device.caps().matmul_tile_size;
            [(n as u32).div_ceil(wg), (m as u32).div_ceil(wg), b as u32]
        }
    }
}

fn matmul_shader_source(kernel: MatMulKernel) -> &'static str {
    match kernel {
        MatMulKernel::GemvF32 | MatMulKernel::GemvColF32 => MATMUL_GEMV,
        MatMulKernel::NaiveF32 => MATMUL_NAIVE,
        MatMulKernel::TiledF32 => MATMUL_TILED,
        MatMulKernel::TiledVecF32 => MATMUL_TILED_VEC,
        MatMulKernel::TiledF16 | MatMulKernel::TiledVecF16 => MATMUL_TILED_F16,
        MatMulKernel::TiledF16AccF32 | MatMulKernel::TiledVecF16AccF32 => MATMUL_TILED_F16ACC,
        MatMulKernel::TiledBf16 | MatMulKernel::TiledVecBf16 => MATMUL_TILED_BF16,
        MatMulKernel::TiledBf16AccF32 | MatMulKernel::TiledVecBf16AccF32 => MATMUL_TILED_BF16ACC,
    }
}

fn matmul_entry_point(kernel: MatMulKernel) -> &'static str {
    match kernel {
        MatMulKernel::GemvF32 => "matmul_gemv_f32",
        MatMulKernel::GemvColF32 => "matmul_gemv_col_f32",
        MatMulKernel::NaiveF32 => "matmul_naive_f32",
        MatMulKernel::TiledF32 => "matmul_tiled_f32",
        MatMulKernel::TiledVecF32 => "matmul_tiled_vec_f32",
        MatMulKernel::TiledF16 => "matmul_tiled_f16",
        MatMulKernel::TiledVecF16 => "matmul_tiled_vec_f16",
        MatMulKernel::TiledF16AccF32 => "matmul_tiled_f16acc",
        MatMulKernel::TiledVecF16AccF32 => "matmul_tiled_vec_f16acc",
        MatMulKernel::TiledBf16 => "matmul_tiled_bf16",
        MatMulKernel::TiledVecBf16 => "matmul_tiled_vec_bf16",
        MatMulKernel::TiledBf16AccF32 => "matmul_tiled_bf16acc",
        MatMulKernel::TiledVecBf16AccF32 => "matmul_tiled_vec_bf16acc",
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

fn binary_entry_point_i32(kernel: &str) -> Option<&'static str> {
    match binary_entry_point(kernel)? {
        "add_f32" => Some("add_i32"),
        "sub_f32" => Some("sub_i32"),
        "mul_f32" => Some("mul_i32"),
        "min_f32" => Some("min_i32"),
        "max_f32" => Some("max_i32"),
        _ => None,
    }
}

fn binary_entry_point_u32(kernel: &str) -> Option<&'static str> {
    match binary_entry_point(kernel)? {
        "add_f32" => Some("add_u32"),
        "sub_f32" => Some("sub_u32"),
        "mul_f32" => Some("mul_u32"),
        "min_f32" => Some("min_u32"),
        "max_f32" => Some("max_u32"),
        _ => None,
    }
}

/// Maximum `head_dim` and `v_dim` supported by fused SDPA WGSL kernels.
pub const MAX_SDPA_DIM: usize = 256;

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

fn reduce_entry_point(op: ReduceOp, dtype: DType) -> Option<String> {
    let base = match op {
        ReduceOp::Sum => "reduce_sum",
        ReduceOp::Max => "reduce_max",
        ReduceOp::Min => "reduce_min",
        ReduceOp::ArgMax => "reduce_argmax",
        ReduceOp::ArgMin => "reduce_argmin",
    };
    Some(format!("{}_{}", base, float_type_suffix(dtype)))
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

/// Packed-bf16 elemwise dispatch (grid-stride loops with per-element atomic RMW).
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
    let kernel = compile_elemwise_kernel(device, source, entry_point)?;
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
    let grid = elemwise_workgroup_count(device, out_layout.shape().elem_count());
    backing.with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
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

struct ElemwiseIntDispatch<'a> {
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

fn dispatch_elemwise_int(args: ElemwiseIntDispatch<'_>) -> Result<()> {
    let ElemwiseIntDispatch {
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
    let kernel = compile_elemwise_kernel(device, source, entry_point)?;
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
    let grid = elemwise_workgroup_count(device, out_layout.shape().elem_count());
    backing.with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
}

fn compile_matmul_kernel(device: &WgpuDevice, kernel: MatMulKernel) -> Result<WgpuKernel> {
    let source = matmul_shader_source(kernel);
    let entry = matmul_entry_point(kernel);
    let caps = device.caps();
    let (tuned, wg) = match kernel {
        MatMulKernel::GemvF32 | MatMulKernel::GemvColF32 => {
            let tuned = tune_matmul_shader_source(
                &super::intel_caps::tune_shader_source(source, caps),
                caps,
            );
            (tuned, caps.elem_workgroup_size)
        }
        _ => (
            tune_matmul_shader_source(source, caps),
            caps.matmul_tile_size,
        ),
    };
    WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, wg)
}

fn compile_reduce_kernel(
    device: &WgpuDevice,
    dtype: DType,
    entry_point: &str,
) -> Result<WgpuKernel> {
    let source = reduce_shader_source(dtype);
    let tuned = super::intel_caps::tune_shader_source(source, device.caps());
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
        if device.caps().supports_bf16_gpu_kernels() {
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

    if dtype == DType::F16 {
        if device.caps().supports_f16_gpu_kernels() {
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::F16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new_affine(
                out_layout.shape().elem_count(),
                &out_layout,
                layout,
                mul,
                add,
            );
            let kernel = compile_elemwise_kernel(&device, UNARY_F16, "affine_f16")?;
            let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
                device.device(),
                device.queue(),
                buffer_offset(&out, &out_layout),
                buffer_offset(storage, layout),
                None,
                uniforms.as_bytes(),
            )?;
            let grid = elemwise_workgroup_count(&device, out_layout.shape().elem_count());
            storage
                .backing()
                .with_unmapped(|| kernel.dispatch_bind_group(&device, &bind_group, [grid, 1, 1]))
                .map_err(Error::from)?;
            return Ok(out);
        }
        let f32 = storage.to_dtype(layout, DType::F32)?;
        let f32_layout = Layout::contiguous(layout.shape());
        let out_f32 = dispatch_affine(&f32, &f32_layout, mul, add)?;
        return out_f32.to_dtype(layout, DType::F16);
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

fn cmp_shader(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => CMP_F16,
        DType::BF16 => CMP_BF16,
        _ => CMP,
    }
}

fn cmp_entry_point(op: CmpOp, dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => match op {
            CmpOp::Eq => "eq_f16",
            CmpOp::Ne => "ne_f16",
            CmpOp::Lt => "lt_f16",
            CmpOp::Le => "le_f16",
            CmpOp::Gt => "gt_f16",
            CmpOp::Ge => "ge_f16",
        },
        DType::BF16 => match op {
            CmpOp::Eq => "eq_bf16",
            CmpOp::Ne => "ne_bf16",
            CmpOp::Lt => "lt_bf16",
            CmpOp::Le => "le_bf16",
            CmpOp::Gt => "gt_bf16",
            CmpOp::Ge => "ge_bf16",
        },
        _ => match op {
            CmpOp::Eq => "eq_f32",
            CmpOp::Ne => "ne_f32",
            CmpOp::Lt => "lt_f32",
            CmpOp::Le => "le_f32",
            CmpOp::Gt => "gt_f32",
            CmpOp::Ge => "ge_f32",
        },
    }
}

pub fn dispatch_cmp(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: CmpOp,
) -> CandleResult<WgpuStorage> {
    let dtype = lhs.dtype();
    if dtype != rhs.dtype() {
        return Err(Error::UnsupportedDTypeForOp(dtype, "cmp").bt());
    }
    let device = lhs.device().clone();
    if (dtype == DType::BF16 || dtype == DType::F16)
        && !device.caps().has_gpu_kernels_for(dtype)
    {
        let lhs_f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
        let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
        let lhs_layout = Layout::contiguous(lhs_layout.shape());
        let rhs_layout = Layout::contiguous(rhs_layout.shape());
        return dispatch_cmp(&lhs_f32, &rhs_f32, &lhs_layout, &rhs_layout, op);
    }
    if dtype != DType::F16 && dtype != DType::BF16 {
        require_f32(dtype, "cmp")?;
    }

    let out_shape = lhs_layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::U8)?;
    let out_layout = Layout::contiguous(out_shape);
    let n = out_layout.shape().elem_count();
    let align = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    out.write_bytes(&vec![0u8; n.div_ceil(align) * align])
        .map_err(Error::from)?;
    let uniforms = KernelUniforms::new(
        out_layout.shape().elem_count(),
        &out_layout,
        lhs_layout,
        Some(rhs_layout),
    );
    let entry = cmp_entry_point(op, dtype);
    let kernel = compile_elemwise_kernel(&device, cmp_shader(dtype), entry)?;
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
    let grid = elemwise_workgroup_count(&device, out_layout.shape().elem_count());
    lhs.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(&device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

pub fn dispatch_powf(
    storage: &WgpuStorage,
    layout: &Layout,
    exp: f64,
) -> CandleResult<WgpuStorage> {
    dispatch_unary_param(
        storage,
        layout,
        exp,
        "powf_f32",
        "powf_f16",
        "powf_bf16",
        "powf",
    )
}

pub fn dispatch_elu(
    storage: &WgpuStorage,
    layout: &Layout,
    alpha: f64,
) -> CandleResult<WgpuStorage> {
    dispatch_unary_param(
        storage, layout, alpha, "elu_f32", "elu_f16", "elu_bf16", "elu",
    )
}

fn dispatch_unary_param(
    storage: &WgpuStorage,
    layout: &Layout,
    param: f64,
    entry_f32: &str,
    entry_f16: &str,
    entry_bf16: &str,
    op: &'static str,
) -> CandleResult<WgpuStorage> {
    let dtype = storage.dtype();
    let device = storage.device().clone();
    if dtype == DType::BF16 {
        if device.caps().supports_bf16_gpu_kernels() {
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new_unary_f32(
                out_layout.shape().elem_count(),
                &out_layout,
                layout,
                param,
            );
            dispatch_elemwise_bf16(ElemwiseBf16Dispatch {
                device: &device,
                source: UNARY_BF16,
                entry_point: entry_bf16,
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
        let out_f32 = dispatch_unary_param(
            &f32,
            &f32_layout,
            param,
            entry_f32,
            entry_f16,
            entry_bf16,
            op,
        )?;
        return out_f32.to_dtype(layout, DType::BF16);
    }

    if dtype == DType::F16 {
        if device.caps().supports_f16_gpu_kernels() {
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::F16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms = KernelUniforms::new_unary_f32(
                out_layout.shape().elem_count(),
                &out_layout,
                layout,
                param,
            );
            let kernel = compile_elemwise_kernel(&device, UNARY_F16, entry_f16)?;
            let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
                device.device(),
                device.queue(),
                buffer_offset(&out, &out_layout),
                buffer_offset(storage, layout),
                None,
                uniforms.as_bytes(),
            )?;
            let grid = elemwise_workgroup_count(&device, out_layout.shape().elem_count());
            storage
                .backing()
                .with_unmapped(|| kernel.dispatch_bind_group(&device, &bind_group, [grid, 1, 1]))
                .map_err(Error::from)?;
            return Ok(out);
        }
        let f32 = storage.to_dtype(layout, DType::F32)?;
        let f32_layout = Layout::contiguous(layout.shape());
        let out_f32 = dispatch_unary_param(
            &f32,
            &f32_layout,
            param,
            entry_f32,
            entry_f16,
            entry_bf16,
            op,
        )?;
        return out_f32.to_dtype(layout, DType::F16);
    }

    require_f32(dtype, op)?;
    let out_shape = layout.shape();
    let out = WgpuStorage::alloc(&device, out_shape, DType::F32)?;
    let out_layout = Layout::contiguous(out_shape);
    let uniforms =
        KernelUniforms::new_unary_f32(out_layout.shape().elem_count(), &out_layout, layout, param);
    let kernel = compile_elemwise_kernel(&device, UNARY, entry_f32)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&out, &out_layout),
        buffer_offset(storage, layout),
        None,
        uniforms.as_bytes(),
    )?;
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
        if device.caps().supports_bf16_gpu_kernels() {
            let entry = unary_entry_point_bf16(B::KERNEL).ok_or_else(|| {
                WgpuError::Message(format!(
                    "wgpu bf16 unary kernel not available for op {}",
                    B::NAME
                ))
            })?;
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms =
                KernelUniforms::new(out_layout.shape().elem_count(), &out_layout, layout, None);
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
        if device.caps().supports_f16_gpu_kernels() {
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
        if device.caps().supports_bf16_gpu_kernels() {
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
        if device.caps().supports_f16_gpu_kernels() {
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

    if dtype == DType::I32 {
        let entry = binary_entry_point_i32(B::KERNEL).ok_or_else(|| {
            WgpuError::Message(format!(
                "wgpu i32 binary kernel not available for op {}",
                B::NAME
            ))
        })?;
        let out_shape = lhs_layout.shape();
        let out = WgpuStorage::alloc(&device, out_shape, DType::I32)?;
        let out_layout = Layout::contiguous(out_shape);
        dispatch_elemwise_int(ElemwiseIntDispatch {
            device: &device,
            source: BINARY_I32,
            entry_point: entry,
            out: &out,
            out_layout: &out_layout,
            in0: lhs,
            in0_layout: lhs_layout,
            in1: Some((rhs, rhs_layout)),
            backing: lhs.backing(),
            uniforms: KernelUniforms::new(
                out_layout.shape().elem_count(),
                &out_layout,
                lhs_layout,
                Some(rhs_layout),
            ),
        })
        .map_err(Error::from)?;
        return Ok(out);
    }

    if dtype == DType::U32 {
        let entry = binary_entry_point_u32(B::KERNEL).ok_or_else(|| {
            WgpuError::Message(format!(
                "wgpu u32 binary kernel not available for op {}",
                B::NAME
            ))
        })?;
        let out_shape = lhs_layout.shape();
        let out = WgpuStorage::alloc(&device, out_shape, DType::U32)?;
        let out_layout = Layout::contiguous(out_shape);
        dispatch_elemwise_int(ElemwiseIntDispatch {
            device: &device,
            source: BINARY_U32,
            entry_point: entry,
            out: &out,
            out_layout: &out_layout,
            in0: lhs,
            in0_layout: lhs_layout,
            in1: Some((rhs, rhs_layout)),
            backing: lhs.backing(),
            uniforms: KernelUniforms::new(
                out_layout.shape().elem_count(),
                &out_layout,
                lhs_layout,
                Some(rhs_layout),
            ),
        })
        .map_err(Error::from)?;
        return Ok(out);
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
        let out_shape = {
            let lhs_dims = lhs_layout.shape().dims();
            let dim = lhs_dims.len();
            let mut out_dims = lhs_dims[..dim - 2].to_vec();
            out_dims.push(m);
            out_dims.push(n);
            Shape::from(out_dims)
        };
        let out_layout = Layout::contiguous(&out_shape);

        if device.caps().supports_shader_f16 {
            let out_f32 =
                dispatch_matmul_inner(lhs, rhs, (b, m, n, k), lhs_layout, rhs_layout, DType::F32)?;
            return out_f32.to_dtype(&out_layout, DType::F16);
        }

        let lhs_f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
        let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
        let lhs_layout = Layout::contiguous(lhs_layout.shape());
        let rhs_layout = Layout::contiguous(rhs_layout.shape());
        let out_f32 = dispatch_matmul_inner(
            &lhs_f32,
            &rhs_f32,
            (b, m, n, k),
            &lhs_layout,
            &rhs_layout,
            DType::F32,
        )?;
        return out_f32.to_dtype(&out_layout, DType::F16);
    }

    if storage_dtype == DType::BF16 && !device.caps().supports_native_bf16() {
        let out_shape = {
            let lhs_dims = lhs_layout.shape().dims();
            let dim = lhs_dims.len();
            let mut out_dims = lhs_dims[..dim - 2].to_vec();
            out_dims.push(m);
            out_dims.push(n);
            Shape::from(out_dims)
        };
        let out_layout = Layout::contiguous(&out_shape);
        let out_f32 =
            dispatch_matmul_inner(lhs, rhs, (b, m, n, k), lhs_layout, rhs_layout, DType::F32)?;
        return out_f32.to_dtype(&out_layout, DType::BF16);
    }

    dispatch_matmul_inner(
        lhs,
        rhs,
        (b, m, n, k),
        lhs_layout,
        rhs_layout,
        storage_dtype,
    )
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
    let kernel_kind = select_matmul_kernel(&device, lhs.dtype(), m, n, k);
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

    let [grid_x, grid_y, grid_z] = matmul_dispatch_grid(kernel_kind, &device, b, m, n);

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

struct QMatMulPaths {
    f32_source: &'static str,
    f32_entry: &'static str,
    f16_source: &'static str,
    f16_entry: &'static str,
}

fn dispatch_qmatmul(
    lhs: &WgpuStorage,
    rhs_buffer: &Arc<wgpu::Buffer>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_layout: &Layout,
    paths: QMatMulPaths,
    k_block: usize,
) -> CandleResult<WgpuStorage> {
    let lhs_dtype = lhs.dtype();
    let device = lhs.device().clone();
    let (source, entry_point) = match lhs_dtype {
        DType::F32 => (paths.f32_source, paths.f32_entry),
        DType::F16 if device.caps().supports_shader_f16 => (paths.f16_source, paths.f16_entry),
        DType::F16 => {
            let f32 = lhs.to_dtype(lhs_layout, DType::F32)?;
            let layout = Layout::contiguous(lhs_layout.shape());
            return dispatch_qmatmul(&f32, rhs_buffer, (b, m, n, k), &layout, paths, k_block);
        }
        other => return Err(Error::UnsupportedDTypeForOp(other, "qmatmul").bt()),
    };
    if !k.is_multiple_of(k_block) {
        return Err(
            WgpuError::Message(format!("qmatmul k={k} must be divisible by {k_block}")).into(),
        );
    }

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
        QMatMulPaths {
            f32_source: QMATMUL_Q4_0,
            f32_entry: "qmatmul_q4_0_f32",
            f16_source: QMATMUL_Q4_0_F16,
            f16_entry: "qmatmul_q4_0_f16",
        },
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
        QMatMulPaths {
            f32_source: QMATMUL_Q5_0,
            f32_entry: "qmatmul_q5_0_f32",
            f16_source: QMATMUL_Q5_0_F16,
            f16_entry: "qmatmul_q5_0_f16",
        },
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
        QMatMulPaths {
            f32_source: QMATMUL_Q8_0,
            f32_entry: "qmatmul_q8_0_f32",
            f16_source: QMATMUL_Q8_0_F16,
            f16_entry: "qmatmul_q8_0_f16",
        },
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
        QMatMulPaths {
            f32_source: QMATMUL_Q4_K,
            f32_entry: "qmatmul_q4_k_f32",
            f16_source: QMATMUL_Q4_K_F16,
            f16_entry: "qmatmul_q4_k_f16",
        },
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
        return Err(Error::Msg(
            "wgpu quantize requires contiguous f32 input".into(),
        ));
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
        other => {
            return Err(Error::Msg(format!(
                "wgpu gpu quant unsupported for {other:?}"
            )))
        }
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
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, source, entry, 1).map_err(Error::from)?;
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
    let dtype = storage.dtype();
    require_float(dtype, "reduce")?;
    let device_ref = storage.device();
    let compute_dtype = if device_ref.caps().has_gpu_kernels_for(dtype) {
        dtype
    } else {
        DType::F32
    };
    if compute_dtype != dtype {
        let cast = storage.to_dtype(layout, compute_dtype)?;
        let cast_layout = Layout::contiguous(layout.shape());
        let out = dispatch_reduce(&cast, op, &cast_layout, reduce_dims)?;
        let mut dst_dims = layout.dims().to_vec();
        for &dim in reduce_dims {
            dst_dims[dim] = 1;
        }
        let out_layout = Layout::contiguous(Shape::from(dst_dims));
        return out.to_dtype(&out_layout, dtype);
    }

    if reduce_dims.len() != 1 {
        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
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
        let mut sorted_dims = reduce_dims.to_vec();
        sorted_dims.sort_unstable();
        let mut current = storage.clone();
        let mut current_layout = layout.clone();
        for &dim in &sorted_dims {
            current = dispatch_reduce(&current, op, &current_layout, &[dim])?;
            let mut dst_dims = current_layout.dims().to_vec();
            dst_dims[dim] = 1;
            current_layout = Layout::contiguous(Shape::from(dst_dims));
        }
        return Ok(current);
    }

    let entry = reduce_entry_point(op, compute_dtype)
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
        let out_dtype = if arg_index { DType::U32 } else { compute_dtype };
        return WgpuStorage::alloc(storage.device(), &dst_shape, out_dtype).map_err(Error::from);
    }
    let reduce_chunk_size = src_elem_count / dst_elem_count;

    let device = storage.device().clone();
    let out = WgpuStorage::alloc(&device, &dst_shape, compute_dtype)?;
    let uniforms = ReduceUniforms::new(
        src_elem_count,
        dst_elem_count,
        reduce_chunk_size,
        reduce_dims[0],
        &dst_layout,
        layout,
    );

    let kernel = compile_reduce_kernel(&device, compute_dtype, &entry)?;
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
            kernel.dispatch_bind_group(&device, &bind_group, per_elem_dispatch_grid(dst_elem_count))
        })
        .map_err(Error::from)?;
    if arg_index {
        return arg_index_f32_to_u32(&out, &dst_layout);
    }
    Ok(out)
}

/// Argmax/argmin kernels write indices as f32; Candle expects `U32` storage.
fn arg_index_f32_to_u32(out_f32: &WgpuStorage, layout: &Layout) -> CandleResult<WgpuStorage> {
    dispatch_cast_f32_u32(out_f32, layout).map_err(Error::from)
}

/// GPU cast from f32 indices (written by argmax/argmin kernels) to u32 storage.
pub fn dispatch_cast_f32_u32(src: &WgpuStorage, layout: &Layout) -> Result<WgpuStorage> {
    let device = src.device();
    let out = WgpuStorage::alloc(device, layout.shape(), DType::U32)?;
    let out_layout = Layout::contiguous(layout.shape());
    let uniforms = KernelUniforms::new(out_layout.shape().elem_count(), &out_layout, layout, None);
    let kernel = WgpuKernel::compile_with_workgroup_size(device, CAST, "cast_f32_u32", 32)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(&out, &out_layout),
        buffer_offset(src, layout),
        None,
        uniforms.as_bytes(),
    )?;
    let elem_count = out_layout.shape().elem_count();
    let grid = workgroup_count(32, elem_count);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(out)
}

fn compile_copy_kernel(device: &WgpuDevice, source: &str, entry: &str) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(source, device.caps());
    WgpuKernel::compile_with_workgroup_size(
        device,
        &tuned,
        entry,
        device.caps().elem_workgroup_size,
    )
}

/// Copy a potentially strided u32/i32 tensor into a contiguous destination buffer.
pub fn dispatch_copy_strided_u32(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    dst_offset: usize,
    src_layout: &Layout,
) -> Result<()> {
    if !matches!(src.dtype(), DType::U32 | DType::I32) || src.dtype() != dst.dtype() {
        return Err(WgpuError::Message(
            "wgpu copy_strided_u32 requires matching u32/i32 dtypes".into(),
        ));
    }
    let device = src.device();
    let elem_count = src_layout.shape().elem_count();
    let dst_shape = Shape::from(elem_count);
    let dst_layout = Layout::new(dst_shape, vec![1], dst_offset);
    let uniforms = KernelUniforms::new(elem_count, &dst_layout, src_layout, None);
    let kernel = compile_copy_kernel(device, COPY_U32, "copy_strided_u32")?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(dst, &dst_layout),
        buffer_offset(src, src_layout),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = elemwise_workgroup_count(device, elem_count);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

pub fn dispatch_copy_strided_src(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    dst_offset: usize,
    src_layout: &Layout,
) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(WgpuError::Message(
            "wgpu copy_strided_src dtype mismatch between src and dst".into(),
        ));
    }
    let dtype = src.dtype();
    require_float(dtype, "copy_strided_src").map_err(|e| WgpuError::Message(e.to_string()))?;
    let device = src.device();
    let compute_dtype = if device.caps().has_gpu_kernels_for(dtype) {
        dtype
    } else {
        DType::F32
    };
    if compute_dtype != dtype {
        let cast_src = src
            .to_dtype(src_layout, compute_dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        let cast_layout = Layout::contiguous(src_layout.shape());
        let mut cast_dst = WgpuStorage::alloc(device, src_layout.shape(), compute_dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        dispatch_copy_strided_src(&cast_src, &mut cast_dst, dst_offset, &cast_layout)?;
        let out = cast_dst
            .to_dtype(&cast_layout, dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        *dst = out;
        return Ok(());
    }

    let elem_count = src_layout.shape().elem_count();
    let dst_shape = Shape::from(elem_count);
    let dst_layout = Layout::new(dst_shape.clone(), vec![1], dst_offset);
    let uniforms = KernelUniforms::new(elem_count, &dst_layout, src_layout, None);
    let entry = format!("copy_strided_{}", float_type_suffix(compute_dtype));
    let kernel = compile_copy_kernel(device, copy_shader_source(compute_dtype), &entry)?;
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        buffer_offset(dst, &dst_layout),
        buffer_offset(src, src_layout),
        None,
        uniforms.as_bytes(),
    )?;
    let grid = elemwise_workgroup_count(device, elem_count);
    src.backing()
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
    if src.dtype() != dst.dtype() {
        return Err(WgpuError::Message(
            "wgpu copy2d dtype mismatch between src and dst".into(),
        ));
    }
    let dtype = src.dtype();
    require_float(dtype, "copy2d").map_err(|e| WgpuError::Message(e.to_string()))?;
    let device = src.device();
    let compute_dtype = if device.caps().has_gpu_kernels_for(dtype) {
        dtype
    } else {
        DType::F32
    };
    if compute_dtype != dtype {
        let src_layout = Layout::contiguous(Shape::from(src.elem_count()));
        let dst_layout = Layout::contiguous(Shape::from(dst.elem_count()));
        let cast_src = src
            .to_dtype(&src_layout, compute_dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        let mut cast_dst = dst
            .to_dtype(&dst_layout, compute_dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        dispatch_copy2d(&cast_src, &mut cast_dst, params)?;
        let out = cast_dst
            .to_dtype(&dst_layout, dtype)
            .map_err(|e| WgpuError::Message(e.to_string()))?;
        *dst = out;
        return Ok(());
    }
    let uniforms = Copy2dUniforms {
        d1: params.d1 as u32,
        d2: params.d2 as u32,
        src_stride: params.src_stride as u32,
        dst_stride: params.dst_stride as u32,
        src_offset: params.src_offset as u32,
        dst_offset: params.dst_offset as u32,
        _pad: [0; 66],
    };
    let entry = format!("copy2d_{}", float_type_suffix(compute_dtype));
    let kernel = WgpuKernel::compile_with_workgroup_size(
        device,
        copy2d_shader_source(compute_dtype),
        &entry,
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
    let total = params.d1 * params.d2;
    let grid = elemwise_workgroup_count(device, total);
    dst.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

fn rms_norm_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (RMS_NORM_F16, "rms_norm_f16"),
        DType::BF16 => (RMS_NORM_BF16, "rms_norm_bf16"),
        _ => (RMS_NORM, "rms_norm_f32"),
    }
}

fn layer_norm_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (LAYER_NORM_F16, "layer_norm_f16"),
        DType::BF16 => (LAYER_NORM_BF16, "layer_norm_bf16"),
        _ => (LAYER_NORM, "layer_norm_f32"),
    }
}

fn rope_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (ROPE_F16, "rope_f16"),
        DType::BF16 => (ROPE_BF16, "rope_bf16"),
        _ => (ROPE, "rope_f32"),
    }
}

fn rope_i_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (ROPE_I_F16, "rope_i_f16"),
        DType::BF16 => (ROPE_I_BF16, "rope_i_bf16"),
        _ => (ROPE_I, "rope_i_f32"),
    }
}

fn rope_thd_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (ROPE_THD_F16, "rope_thd_f16"),
        DType::BF16 => (ROPE_THD_BF16, "rope_thd_bf16"),
        _ => (ROPE_THD, "rope_thd_f32"),
    }
}

fn sdpa_vector_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (SDPA_VECTOR_F16, "sdpa_vector_f16"),
        DType::BF16 => (SDPA_VECTOR_BF16, "sdpa_vector_bf16"),
        _ => (SDPA_VECTOR, "sdpa_vector_f32"),
    }
}

fn sdpa_full_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (SDPA_FULL_F16, "sdpa_full_f16"),
        DType::BF16 => (SDPA_FULL_BF16, "sdpa_full_bf16"),
        _ => (SDPA_FULL, "sdpa_full_f32"),
    }
}

/// RMS normalization with dtype-native kernels (f32 / f16 / bf16).
pub fn dispatch_rms_norm(
    x: &WgpuStorage,
    alpha: &WgpuStorage,
    x_layout: &Layout,
    alpha_layout: &Layout,
    eps: f32,
) -> CandleResult<WgpuStorage> {
    let dtype = x.dtype();
    require_float(dtype, "rms_norm")?;
    if alpha.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rms_norm alpha dtype {:?} must match x dtype {dtype:?}",
            alpha.dtype()
        )));
    }
    let device = x.device();
    let dims = x_layout.dims();
    let n_cols = *dims
        .last()
        .ok_or_else(|| Error::Msg("empty tensor in rms_norm".into()))?;
    let n_rows = x_layout.shape().elem_count() / n_cols;
    let out = WgpuStorage::alloc(device, x_layout.shape(), dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(x_layout.shape());
    let uniforms = RmsNormUniforms {
        n_rows: n_rows as u32,
        n_cols: n_cols as u32,
        eps_bits: eps.to_bits(),
        _pad: [0; 69],
    };
    let (shader, entry) = rms_norm_shader(dtype);
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, shader, entry, 32).map_err(Error::from)?;
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

/// RMS normalization (f32). Prefer [`dispatch_rms_norm`] for dtype-aware dispatch.
pub fn dispatch_rms_norm_f32(
    x: &WgpuStorage,
    alpha: &WgpuStorage,
    x_layout: &Layout,
    alpha_layout: &Layout,
    eps: f32,
) -> CandleResult<WgpuStorage> {
    dispatch_rms_norm(x, alpha, x_layout, alpha_layout, eps)
}

/// Layer normalization with dtype-native kernels (f32 / f16 / bf16).
pub fn dispatch_layer_norm(
    x: &WgpuStorage,
    alpha: &WgpuStorage,
    beta: &WgpuStorage,
    x_layout: &Layout,
    alpha_layout: &Layout,
    beta_layout: &Layout,
    eps: f32,
) -> CandleResult<WgpuStorage> {
    let dtype = x.dtype();
    require_float(dtype, "layer_norm")?;
    if alpha.dtype() != dtype || beta.dtype() != dtype {
        return Err(Error::Msg(format!(
            "layer_norm alpha/beta dtype must match x {dtype:?}, got alpha={:?} beta={:?}",
            alpha.dtype(),
            beta.dtype()
        )));
    }
    if !(x_layout.is_contiguous() && alpha_layout.is_contiguous() && beta_layout.is_contiguous()) {
        return Err(Error::Msg("Non contiguous layer_norm is not implemented".into()).bt());
    }
    let device = x.device();
    let dims = x_layout.dims();
    let n_cols = *dims
        .last()
        .ok_or_else(|| Error::Msg("empty tensor in layer_norm".into()))?;
    let n_rows = x_layout.shape().elem_count() / n_cols;
    let out = WgpuStorage::alloc(device, x_layout.shape(), dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(x_layout.shape());
    let uniforms = RmsNormUniforms {
        n_rows: n_rows as u32,
        n_cols: n_cols as u32,
        eps_bits: eps.to_bits(),
        _pad: [0; 69],
    };
    let (shader, entry) = layer_norm_shader(dtype);
    let kernel = WgpuKernel::compile_extended(device, shader, entry, 32).map_err(Error::from)?;
    let bind_group = kernel
        .create_extended_bind_group(
            device,
            buffer_offset(&out, &out_layout),
            buffer_offset(x, x_layout),
            buffer_offset(alpha, alpha_layout),
            buffer_offset(beta, beta_layout),
            uniforms.as_bytes(),
        )
        .map_err(Error::from)?;
    x.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [n_rows as u32, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

/// Layer normalization (f32). Prefer [`dispatch_layer_norm`] for dtype-aware dispatch.
pub fn dispatch_layer_norm_f32(
    x: &WgpuStorage,
    alpha: &WgpuStorage,
    beta: &WgpuStorage,
    x_layout: &Layout,
    alpha_layout: &Layout,
    beta_layout: &Layout,
    eps: f32,
) -> CandleResult<WgpuStorage> {
    dispatch_layer_norm(x, alpha, beta, x_layout, alpha_layout, beta_layout, eps)
}

/// Sigmoid activation using dtype-native unary kernels.
pub fn dispatch_sigmoid(storage: &WgpuStorage, layout: &Layout) -> CandleResult<WgpuStorage> {
    dispatch_unary_kernel(storage, layout, "usigmoid", "sigmoid")
}

fn dispatch_unary_kernel(
    storage: &WgpuStorage,
    layout: &Layout,
    kernel: &str,
    op: &'static str,
) -> CandleResult<WgpuStorage> {
    let dtype = storage.dtype();
    let device = storage.device().clone();
    if dtype == DType::BF16 {
        if device.caps().supports_bf16_gpu_kernels() {
            let entry = unary_entry_point_bf16(kernel).ok_or_else(|| {
                WgpuError::Message(format!("wgpu bf16 unary kernel not available for op {op}"))
            })?;
            let out_shape = layout.shape();
            let out = WgpuStorage::alloc(&device, out_shape, DType::BF16)?;
            let out_layout = Layout::contiguous(out_shape);
            let uniforms =
                KernelUniforms::new(out_layout.shape().elem_count(), &out_layout, layout, None);
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
        let out_f32 = dispatch_unary_kernel(&f32, &f32_layout, kernel, op)?;
        return out_f32.to_dtype(layout, DType::BF16);
    }

    if dtype == DType::F16 {
        if device.caps().supports_f16_gpu_kernels() {
            let entry = unary_entry_point_f16(kernel).ok_or_else(|| {
                WgpuError::Message(format!("wgpu f16 unary kernel not available for op {op}"))
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
        let out_f32 = dispatch_unary_kernel(&f32, &f32_layout, kernel, op)?;
        return out_f32.to_dtype(layout, DType::F16);
    }

    require_f32(dtype, op)?;
    let entry = unary_entry_point(kernel).ok_or_else(|| {
        WgpuError::Message(format!("wgpu unary kernel not available for op {op}"))
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

/// NeoX-style rotary positional embedding with dtype-native kernels.
pub fn dispatch_rope(
    src: &WgpuStorage,
    cos: &WgpuStorage,
    sin: &WgpuStorage,
    src_layout: &Layout,
    cos_layout: &Layout,
    sin_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    let dtype = src.dtype();
    require_float(dtype, "rope")?;
    if cos.dtype() != dtype || sin.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rope cos/sin dtype must match src {dtype:?}, got cos={:?} sin={:?}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let (b, h, t, d) = src_layout.shape().dims4()?;
    let out = WgpuStorage::alloc(src.device(), src_layout.shape(), dtype).map_err(Error::from)?;
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
    let (shader, entry) = rope_shader(dtype);
    let kernel = WgpuKernel::compile_extended(device, shader, entry, 32).map_err(Error::from)?;
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

/// Rotary positional embedding (f32). Prefer [`dispatch_rope`] for dtype-aware dispatch.
pub fn dispatch_rope_f32(
    src: &WgpuStorage,
    cos: &WgpuStorage,
    sin: &WgpuStorage,
    src_layout: &Layout,
    cos_layout: &Layout,
    sin_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_rope(src, cos, sin, src_layout, cos_layout, sin_layout)
}

/// Interleaved rotary positional embedding (pairs adjacent elements in head dim).
pub fn dispatch_rope_i(
    src: &WgpuStorage,
    cos: &WgpuStorage,
    sin: &WgpuStorage,
    src_layout: &Layout,
    cos_layout: &Layout,
    sin_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    let dtype = src.dtype();
    require_float(dtype, "rope_i")?;
    if cos.dtype() != dtype || sin.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rope_i cos/sin dtype must match src {dtype:?}, got cos={:?} sin={:?}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    if !(src_layout.is_contiguous() && cos_layout.is_contiguous() && sin_layout.is_contiguous()) {
        return Err(Error::Msg("Non contiguous rope_i is not implemented".into()).bt());
    }
    let (b, h, t, d) = src_layout.shape().dims4()?;
    let out = WgpuStorage::alloc(src.device(), src_layout.shape(), dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(src_layout.shape());
    let stride_b = if cos_layout.dims().len() == 3 && sin_layout.dims().len() == 3 {
        (h * t * d) as u32
    } else {
        0u32
    };
    let uniforms = RopeIUniforms {
        bh: (b * h) as u32,
        td: (t * d) as u32,
        stride_b,
        _pad: [0; 68],
    };
    let device = src.device();
    let (shader, entry) = rope_i_shader(dtype);
    let kernel = WgpuKernel::compile_extended(device, shader, entry, 32).map_err(Error::from)?;
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
    let grid = workgroup_count(32, b * h * t * d / 2);
    src.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

/// THD-layout rotary positional embedding: tensor shape `(b, t, h, d)`.
pub fn dispatch_rope_thd(
    src: &WgpuStorage,
    cos: &WgpuStorage,
    sin: &WgpuStorage,
    src_layout: &Layout,
    cos_layout: &Layout,
    sin_layout: &Layout,
) -> CandleResult<WgpuStorage> {
    let dtype = src.dtype();
    require_float(dtype, "rope_thd")?;
    if cos.dtype() != dtype || sin.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rope_thd cos/sin dtype must match src {dtype:?}, got cos={:?} sin={:?}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    if !(src_layout.is_contiguous() && cos_layout.is_contiguous() && sin_layout.is_contiguous()) {
        return Err(Error::Msg("Non contiguous rope_thd is not implemented".into()).bt());
    }
    let (b, t, h, d) = src_layout.shape().dims4()?;
    let out = WgpuStorage::alloc(src.device(), src_layout.shape(), dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(src_layout.shape());
    let stride_b = if cos_layout.dims().len() == 3 && sin_layout.dims().len() == 3 {
        (h * t * d) as u32
    } else {
        0u32
    };
    let uniforms = RopeThdUniforms {
        b: b as u32,
        t: t as u32,
        h: h as u32,
        d: d as u32,
        stride_b,
        _pad: [0; 67],
    };
    let device = src.device();
    let (shader, entry) = rope_thd_shader(dtype);
    let kernel = WgpuKernel::compile_extended(device, shader, entry, 32).map_err(Error::from)?;
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
    let grid = workgroup_count(32, b * t * h * d / 2);
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
        return Err(Error::UnsupportedDTypeForOp(cond.dtype(), "where_cond predicate").bt());
    }
    let branch_dtype = on_true.dtype();
    if branch_dtype != on_false.dtype() {
        return Err(Error::UnsupportedDTypeForOp(branch_dtype, "where_cond").bt());
    }
    let device = cond.device();
    if branch_dtype == DType::F16 && !device.caps().supports_f16_gpu_kernels() {
        let t = on_true.to_dtype(on_true_layout, DType::F32)?;
        let f = on_false.to_dtype(on_false_layout, DType::F32)?;
        let t_layout = Layout::contiguous(on_true_layout.shape());
        let f_layout = Layout::contiguous(on_false_layout.shape());
        let out_f32 = dispatch_where_u8_f32(cond, &t, &f, cond_layout, &t_layout, &f_layout)?;
        return out_f32.to_dtype(cond_layout, DType::F16);
    }
    if branch_dtype == DType::BF16 && !device.caps().supports_bf16_gpu_kernels() {
        let t = on_true.to_dtype(on_true_layout, DType::F32)?;
        let f = on_false.to_dtype(on_false_layout, DType::F32)?;
        let t_layout = Layout::contiguous(on_true_layout.shape());
        let f_layout = Layout::contiguous(on_false_layout.shape());
        let out_f32 = dispatch_where_u8_f32(cond, &t, &f, cond_layout, &t_layout, &f_layout)?;
        return out_f32.to_dtype(cond_layout, DType::BF16);
    }
    let (out_dtype, shader, entry) = match branch_dtype {
        DType::F32 => (DType::F32, WHERE_COND, "where_u8_f32"),
        DType::F16 => (DType::F16, WHERE_COND_F16, "where_u8_f16"),
        DType::BF16 => (DType::BF16, WHERE_COND_BF16, "where_u8_bf16"),
        other => return Err(Error::UnsupportedDTypeForOp(other, "where_cond").bt()),
    };
    let elem_count = cond_layout.shape().elem_count();
    let out = WgpuStorage::alloc(device, cond_layout.shape(), out_dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(cond_layout.shape());
    let uniforms = WhereUniforms {
        elem_count: elem_count as u32,
        _pad: [0; 71],
    };
    let kernel = WgpuKernel::compile_extended(device, shader, entry, 32).map_err(Error::from)?;
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
    let grid = workgroup_count(32, elem_count);
    cond.backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))
        .map_err(Error::from)?;
    Ok(out)
}

fn softmax_shader(dtype: DType) -> (&'static str, &'static str) {
    match dtype {
        DType::F16 => (SOFTMAX_F16, "softmax_last_dim_f16"),
        DType::BF16 => (SOFTMAX_BF16, "softmax_last_dim_bf16"),
        _ => (SOFTMAX, "softmax_last_dim_f32"),
    }
}

/// Fused softmax along the last dimension (contiguous f32 / f16 / bf16 input).
pub fn dispatch_softmax_last_dim(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<WgpuStorage> {
    let dtype = storage.dtype();
    require_float(dtype, "softmax")?;
    if !layout.is_contiguous() {
        return Err(Error::Msg("softmax requires contiguous input".into()));
    }
    let dims = layout.dims();
    let last_dim = *dims
        .last()
        .ok_or_else(|| Error::Msg("empty tensor in softmax".into()))?;
    let n_rows = layout.shape().elem_count() / last_dim;
    let device = storage.device();
    let out = WgpuStorage::alloc(device, layout.shape(), dtype).map_err(Error::from)?;
    let out_layout = Layout::contiguous(layout.shape());
    let uniforms = SoftmaxUniforms {
        n_rows: n_rows as u32,
        last_dim: last_dim as u32,
        _pad: [0; 70],
    };
    let (shader, entry) = softmax_shader(dtype);
    let tuned = super::intel_caps::tune_shader_source(shader, device.caps());
    let kernel =
        WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, 32).map_err(Error::from)?;
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

/// Fused softmax along the last dimension (f32). Prefer [`dispatch_softmax_last_dim`].
pub fn dispatch_softmax_last_dim_f32(
    storage: &WgpuStorage,
    layout: &Layout,
) -> CandleResult<WgpuStorage> {
    dispatch_softmax_last_dim(storage, layout)
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
        return Err(Error::Msg(
            "sdpa n_heads must be a multiple of n_kv_heads".into(),
        ));
    }
    if head_dim > MAX_SDPA_DIM {
        return Err(Error::Msg(format!(
            "sdpa supports head_dim <= {MAX_SDPA_DIM}, got {head_dim}"
        )));
    }
    if v_dim > MAX_SDPA_DIM {
        return Err(Error::Msg(format!(
            "sdpa supports v_dim <= {MAX_SDPA_DIM}, got {v_dim}"
        )));
    }

    Ok((bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim))
}

fn require_matching_sdpa_dtype(q: DType, k: DType, v: DType) -> CandleResult<DType> {
    require_float(q, "sdpa")?;
    if k != q || v != q {
        return Err(Error::Msg(format!(
            "sdpa q/k/v dtypes must match, got q={q:?} k={k:?} v={v:?}"
        )));
    }
    Ok(q)
}

/// Fused scaled dot-product attention with vector/full routing (f32 / f16 / bf16).
pub fn dispatch_sdpa(
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
    let dtype = require_matching_sdpa_dtype(q.dtype(), k.dtype(), v.dtype())?;

    let (bs, n_q_heads, _n_kv_heads, q_seq, k_seq, _head_dim, _v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;

    if q_seq > k_seq {
        return Err(Error::Msg("sdpa requires q_seq <= k_seq".into()));
    }

    let use_vector = q_seq <= 8 && mask.is_none() && !do_causal;
    if use_vector {
        return dispatch_sdpa_vector(q, k, v, q_layout, k_layout, v_layout, scale, softcapping);
    }

    if softcapping != 1.0 {
        return Err(Error::Msg(
            "wgpu sdpa_full does not support softcapping (must be 1.0)".into(),
        ));
    }

    if let Some((mask_storage, mask_layout)) = mask {
        if mask_storage.dtype() != dtype {
            return Err(Error::Msg(format!(
                "sdpa mask dtype {:?} must match q dtype {dtype:?}",
                mask_storage.dtype()
            )));
        }
        if !mask_layout.is_contiguous() {
            return Err(Error::Msg("sdpa mask must be contiguous".into()));
        }
        let mask_dims = mask_layout.shape().dims4()?;
        if mask_dims != (bs, n_q_heads, q_seq, k_seq) {
            return Err(Error::Msg(format!(
                "sdpa mask shape must be ({bs}, {n_q_heads}, {q_seq}, {k_seq}), got {mask_dims:?}"
            )));
        }
    }

    dispatch_sdpa_full(
        q, k, v, q_layout, k_layout, v_layout, mask, do_causal, scale,
    )
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
    dispatch_sdpa(
        q,
        k,
        v,
        q_layout,
        k_layout,
        v_layout,
        mask,
        do_causal,
        scale,
        softcapping,
    )
}

/// Fused scaled dot-product attention (prefill/full path).
pub fn dispatch_sdpa_full(
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
    let dtype = require_matching_sdpa_dtype(q.dtype(), k.dtype(), v.dtype())?;
    let (bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;

    let out_shape = Shape::from((bs, n_q_heads, q_seq, v_dim));
    let device = q.device();
    let out = WgpuStorage::alloc(device, &out_shape, dtype).map_err(Error::from)?;
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

    let (shader, entry) = sdpa_full_shader(dtype);
    let kernel = WgpuKernel::compile_sdpa(device, shader, entry, SDPA_WORKGROUP_SIZE)
        .map_err(Error::from)?;
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
    dispatch_sdpa_full(
        q, k, v, q_layout, k_layout, v_layout, mask, do_causal, scale,
    )
}

/// Fused scaled dot-product attention (vector/decode path).
pub fn dispatch_sdpa_vector(
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    q_layout: &Layout,
    k_layout: &Layout,
    v_layout: &Layout,
    scale: f32,
    softcapping: f32,
) -> CandleResult<WgpuStorage> {
    let dtype = require_matching_sdpa_dtype(q.dtype(), k.dtype(), v.dtype())?;
    let (bs, n_q_heads, n_kv_heads, q_seq, k_seq, head_dim, v_dim) =
        validate_sdpa_shapes(q_layout, k_layout, v_layout)?;
    if q_seq > 8 {
        return Err(Error::Msg(format!(
            "sdpa_vector supports q_seq <= 8, got {q_seq}"
        )));
    }

    let out_shape = Shape::from((bs, n_q_heads, q_seq, v_dim));
    let device = q.device();
    let out = WgpuStorage::alloc(device, &out_shape, dtype).map_err(Error::from)?;
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
    let (shader, entry) = sdpa_vector_shader(dtype);
    let kernel = WgpuKernel::compile_extended(device, shader, entry, SDPA_WORKGROUP_SIZE)
        .map_err(Error::from)?;
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
    dispatch_sdpa_vector(q, k, v, q_layout, k_layout, v_layout, scale, softcapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuStorage;

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
        assert_eq!(
            matmul_entry_point(MatMulKernel::TiledVecF32),
            "matmul_tiled_vec_f32"
        );
        assert_eq!(
            matmul_entry_point(MatMulKernel::TiledVecF16),
            "matmul_tiled_vec_f16"
        );
        assert_eq!(
            matmul_entry_point(MatMulKernel::TiledF16AccF32),
            "matmul_tiled_f16acc"
        );
        assert_eq!(
            matmul_shader_source(MatMulKernel::TiledF16),
            MATMUL_TILED_F16
        );
        assert_eq!(
            matmul_shader_source(MatMulKernel::TiledF16AccF32),
            MATMUL_TILED_F16ACC
        );
        assert_eq!(
            matmul_entry_point(MatMulKernel::TiledBf16AccF32),
            "matmul_tiled_bf16acc"
        );
        assert_eq!(
            matmul_shader_source(MatMulKernel::TiledBf16AccF32),
            MATMUL_TILED_BF16ACC
        );
    }

    #[test]
    fn reduce_entry_points() {
        assert_eq!(
            reduce_entry_point(ReduceOp::Sum, DType::F32),
            Some("reduce_sum_f32".to_string())
        );
        assert_eq!(
            reduce_entry_point(ReduceOp::Max, DType::F16),
            Some("reduce_max_f16".to_string())
        );
        assert_eq!(unary_entry_point("ugelu"), Some("gelu_f32"));
        assert_eq!(
            reduce_entry_point(ReduceOp::Min, DType::BF16),
            Some("reduce_min_bf16".to_string())
        );
        assert_eq!(
            reduce_entry_point(ReduceOp::ArgMax, DType::F32),
            Some("reduce_argmax_f32".to_string())
        );
        assert_eq!(
            reduce_entry_point(ReduceOp::ArgMin, DType::F16),
            Some("reduce_argmin_f16".to_string())
        );
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
        let storage =
            WgpuStorage::from_cpu(&device, &CpuStorage::F32(vec![1.0, -2.5, 0.0, 3.25])).unwrap();
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
        WgpuKernel::compile_with_workgroup_size(&device, SOFTMAX_BF16, "softmax_last_dim_bf16", 32)
            .unwrap();
        WgpuKernel::compile_extended(&device, SDPA_VECTOR, "sdpa_vector_f32", SDPA_WORKGROUP_SIZE)
            .unwrap();
        WgpuKernel::compile_extended(
            &device,
            SDPA_VECTOR_BF16,
            "sdpa_vector_bf16",
            SDPA_WORKGROUP_SIZE,
        )
        .unwrap();
        WgpuKernel::compile_sdpa(&device, SDPA_FULL, "sdpa_full_f32", SDPA_WORKGROUP_SIZE).unwrap();
        WgpuKernel::compile_sdpa(
            &device,
            SDPA_FULL_BF16,
            "sdpa_full_bf16",
            SDPA_WORKGROUP_SIZE,
        )
        .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, RMS_NORM_BF16, "rms_norm_bf16", 32)
            .unwrap();
        WgpuKernel::compile_extended(&device, LAYER_NORM, "layer_norm_f32", 32).unwrap();
        WgpuKernel::compile_extended(&device, LAYER_NORM_BF16, "layer_norm_bf16", 32).unwrap();
        WgpuKernel::compile_extended(&device, ROPE_BF16, "rope_bf16", 32).unwrap();
        WgpuKernel::compile_extended(&device, ROPE_I, "rope_i_f32", 32).unwrap();
        WgpuKernel::compile_extended(&device, ROPE_THD, "rope_thd_f32", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(
            &device,
            crate::wgsl::ARGSORT,
            "asort_asc_f32",
            256,
        )
        .unwrap();
        WgpuKernel::compile_with_workgroup_size(
            &device,
            crate::wgsl::ARGSORT_BF16,
            "asort_asc_bf16",
            256,
        )
        .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, QMATMUL_Q5_0, "qmatmul_q5_0_f32", 8)
            .unwrap();
        compile_matmul_kernel(&device, MatMulKernel::TiledBf16).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, UNARY_BF16, "gelu_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, UNARY_BF16, "affine_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_BF16, "add_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_BF16, "min_bf16", 1).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_I32, "add_i32", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, BINARY_U32, "add_u32", 32).unwrap();
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
        WgpuKernel::compile_with_workgroup_size(&device, COPY_BF16, "copy_strided_bf16", 32)
            .unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, COPY_U32, "copy_strided_u32", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, CAST, "cast_f32_u32", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, CAST, "cast_f32_f16", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, CAST, "cast_f16_f32", 32).unwrap();
        WgpuKernel::compile_with_workgroup_size(&device, COPY2D_BF16, "copy2d_bf16", 32).unwrap();
        compile_reduce_kernel(&device, DType::BF16, "reduce_sum_bf16").unwrap();
        WgpuKernel::compile_with_workgroup_size(
            &device,
            crate::wgsl::POOL2D_BF16,
            "avg_pool2d_bf16",
            32,
        )
        .unwrap();
    }
}
