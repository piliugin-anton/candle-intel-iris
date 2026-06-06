// Element-wise unary kernels (packed bf16) with grid-stride loops.
// Per-element atomic RMW handles packed-word writes safely.
//
// Entry points: neg_bf16, exp_bf16, log_bf16, sqrt_bf16, abs_bf16, relu_bf16,
//               recip_bf16, silu_bf16, sigmoid_bf16, gelu_bf16, gelu_erf_bf16,
//               sin_bf16, cos_bf16, tanh_bf16, sqr_bf16, erf_bf16, ceil_bf16,
//               floor_bf16, round_bf16, sign_bf16, affine_bf16

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
const GELU_SQRT_2_OVER_PI: f32 = 0.7978845608028654;

fn sign_bf16_val(x: f32) -> f32 {
    return select(0.0, 1.0, x > 0.0) - select(0.0, 1.0, x < 0.0);
}

fn gelu_approx(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(GELU_SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

@compute @workgroup_size(WG_SIZE)
fn neg_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, -load_bf16_in0(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn exp_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, exp(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn log_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, log(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sqrt_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, sqrt(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn abs_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, abs(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn relu_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, max(x, 0.0));
    }
}

@compute @workgroup_size(WG_SIZE)
fn recip_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, 1.0 / load_bf16_in0(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn silu_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, x / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sigmoid_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, 1.0 / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn gelu_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, gelu_approx(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn gelu_erf_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, 0.5 * x * (1.0 + erf_approx(x * INV_SQRT_2)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sin_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, sin(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn cos_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, cos(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn tanh_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, tanh(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sqr_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, x * x);
    }
}

@compute @workgroup_size(WG_SIZE)
fn erf_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, erf_approx(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn ceil_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, ceil(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn floor_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, floor(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn round_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, round(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sign_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, sign_bf16_val(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn affine_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let mul = bitcast<f32>(kernel_params._pad0);
    let add = bitcast<f32>(kernel_params._pad1);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, load_bf16_in0(i) * mul + add);
    }
}

@compute @workgroup_size(WG_SIZE)
fn powf_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let exp = bitcast<f32>(kernel_params._pad0);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = f32(load_bf16_in0(i));
        var y: f32;
        if (x >= 0.0) {
            y = pow(x, exp);
        } else {
            let exp_round = round(exp);
            if (abs(exp - exp_round) > 1e-6) {
                y = pow(x, exp);
            } else {
                let mag = pow(-x, exp);
                let exp_i = i32(exp_round);
                if ((exp_i & 1) == 0) {
                    y = mag;
                } else {
                    y = -mag;
                }
            }
        }
        store_bf16_out(i, y);
    }
}

@compute @workgroup_size(WG_SIZE)
fn elu_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let alpha = bitcast<f32>(kernel_params._pad0);
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        let x = f32(load_bf16_in0(i));
        let y = select(alpha * (exp(x) - 1.0), x, x > 0.0);
        store_bf16_out(i, y);
    }
}
