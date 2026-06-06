// Element-wise unary kernels (f16) with grid-stride loops and strided/broadcast layouts.
//
// Entry points: neg_f16, exp_f16, log_f16, sqrt_f16, abs_f16, relu_f16,
//               recip_f16, silu_f16, sigmoid_f16, gelu_f16, gelu_erf_f16,
//               sin_f16, cos_f16, tanh_f16, sqr_f16, erf_f16, ceil_f16,
//               floor_f16, round_f16, sign_f16, affine_f16, powf_f16, elu_f16

// Abramowitz & Stegun 7.1.26 — matches libm erf within ~1e-7 for typical ranges.
fn erf_approx(x: f16) -> f16 {
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

const INV_SQRT_2: f16 = 0.7071067811865476;

fn sign_f16_val(x: f16) -> f16 {
    return select(0.0, 1.0, x > 0.0) - select(0.0, 1.0, x < 0.0);
}

// Tanh approximation (matches Candle `Gelu` / PyTorch `approximate='tanh'`).
const GELU_SQRT_2_OVER_PI: f16 = 0.7978845608028654;

fn gelu_approx(x: f16) -> f16 {
    return 0.5 * x * (1.0 + tanh(GELU_SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

@compute @workgroup_size(WG_SIZE)
fn neg_f16(
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
fn exp_f16(
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
fn log_f16(
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
fn sqrt_f16(
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
fn abs_f16(
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
fn relu_f16(
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
fn recip_f16(
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
fn silu_f16(
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
fn sigmoid_f16(
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
fn gelu_f16(
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
fn gelu_erf_f16(
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
fn sin_f16(
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
fn cos_f16(
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
fn tanh_f16(
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
fn sqr_f16(
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
fn erf_f16(
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
fn ceil_f16(
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
fn floor_f16(
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
fn round_f16(
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
fn sign_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, sign_f16_val(load_in0(i)));
    }
}

// `_pad0` / `_pad1` hold `mul` / `add` as f16 bit patterns (see `KernelUniforms::new_affine`).
@compute @workgroup_size(WG_SIZE)
fn affine_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let mul = f16(bitcast<f32>(kernel_params._pad0));
    let add = f16(bitcast<f32>(kernel_params._pad1));
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) * mul + add);
    }
}

@compute @workgroup_size(WG_SIZE)
fn powf_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let exp = f16(bitcast<f32>(kernel_params._pad0));
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = f32(load_in0(i));
        let e = f32(exp);
        var y: f32;
        if (x >= 0.0) {
            y = pow(x, e);
        } else {
            let exp_round = round(e);
            if (abs(e - exp_round) > 1e-6) {
                y = pow(x, e);
            } else {
                let mag = pow(-x, e);
                let exp_i = i32(exp_round);
                if ((exp_i & 1) == 0) {
                    y = mag;
                } else {
                    y = -mag;
                }
            }
        }
        store_out(i, f16(y));
    }
}

@compute @workgroup_size(WG_SIZE)
fn elu_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let alpha = f16(bitcast<f32>(kernel_params._pad0));
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_in0(i);
        store_out(i, select(alpha * (exp(x) - 1.0), x, x > 0.0));
    }
}
