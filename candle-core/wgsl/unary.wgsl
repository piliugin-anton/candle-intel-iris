// Element-wise unary kernels (f32) with grid-stride loops and strided/broadcast layouts.
//
// Entry points: neg_f32, exp_f32, log_f32, sqrt_f32, abs_f32, relu_f32,
//               recip_f32, silu_f32, sigmoid_f32, gelu_f32, gelu_erf_f32,
//               sin_f32, cos_f32, tanh_f32, sqr_f32, erf_f32, ceil_f32,
//               floor_f32, round_f32, sign_f32, affine_f32, powf_f32, elu_f32

// Abramowitz & Stegun 7.1.26 — matches libm erf within ~1e-7 for typical ranges.
fn erf_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * exp(-ax * ax));
    return sign * y;
}

const INV_SQRT_2: f32 = 0.7071067811865476;

fn sign_f32_val(x: f32) -> f32 {
    return select(0.0, 1.0, x > 0.0) - select(0.0, 1.0, x < 0.0);
}

// Tanh approximation (matches Candle `Gelu` / PyTorch `approximate='tanh'`).
const GELU_SQRT_2_OVER_PI: f32 = 0.7978845608028654;

fn gelu_approx(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(GELU_SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

@compute @workgroup_size(WG_SIZE)
fn neg_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, -load_in0(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn exp_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, exp(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn log_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, log(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sqrt_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, sqrt(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn abs_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, abs(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn relu_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, max(x, 0.0));
    }
}

@compute @workgroup_size(WG_SIZE)
fn recip_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, 1.0 / load_in0(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn silu_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, x / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sigmoid_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, 1.0 / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn gelu_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, gelu_approx(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn gelu_erf_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, (erf_approx(x * INV_SQRT_2) + 1.0) * 0.5 * x);
    }
}

@compute @workgroup_size(WG_SIZE)
fn sin_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, sin(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn cos_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, cos(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn tanh_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, tanh(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sqr_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, x * x);
    }
}

@compute @workgroup_size(WG_SIZE)
fn erf_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, erf_approx(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn ceil_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, ceil(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn floor_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, floor(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn round_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, round(load_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sign_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, sign_f32_val(load_in0(i)));
    }
}

// `_pad0` / `_pad1` hold `mul` / `add` as f32 bit patterns (see `KernelUniforms::new_affine`).
@compute @workgroup_size(WG_SIZE)
fn affine_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let mul = bitcast<f32>(kernel_params._pad0);
    let add = bitcast<f32>(kernel_params._pad1);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) * mul + add);
    }
}

// `_pad0` holds the exponent / alpha as an f32 bit pattern.
fn powf_f32_val(x: f32, exp: f32) -> f32 {
    if (x >= 0.0) {
        return pow(x, exp);
    }
    let exp_round = round(exp);
    if (abs(exp - exp_round) > 1e-6) {
        return pow(x, exp);
    }
    let mag = pow(-x, exp);
    let exp_i = i32(exp_round);
    if ((exp_i & 1) == 0) {
        return mag;
    }
    return -mag;
}

@compute @workgroup_size(WG_SIZE)
fn powf_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let exp = bitcast<f32>(kernel_params._pad0);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, powf_f32_val(load_in0(i), exp));
    }
}

@compute @workgroup_size(WG_SIZE)
fn elu_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let alpha = bitcast<f32>(kernel_params._pad0);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, select(alpha * (exp(x) - 1.0), x, x > 0.0));
    }
}
