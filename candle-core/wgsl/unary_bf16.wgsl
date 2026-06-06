// Element-wise unary kernels (packed bf16) with serial execution.
// Two bf16 values share one u32 word; parallel writes would race without atomics.
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

@compute @workgroup_size(1)
fn neg_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, -load_bf16_in0(i));
    }
}

@compute @workgroup_size(1)
fn exp_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, exp(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn log_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, log(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn sqrt_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, sqrt(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn abs_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, abs(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn relu_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, max(x, 0.0));
    }
}

@compute @workgroup_size(1)
fn recip_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, 1.0 / load_bf16_in0(i));
    }
}

@compute @workgroup_size(1)
fn silu_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, x / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(1)
fn sigmoid_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, 1.0 / (1.0 + exp(-x)));
    }
}

@compute @workgroup_size(1)
fn gelu_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, gelu_approx(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn gelu_erf_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, 0.5 * x * (1.0 + erf_approx(x * INV_SQRT_2)));
    }
}

@compute @workgroup_size(1)
fn sin_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, sin(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn cos_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, cos(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn tanh_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, tanh(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn sqr_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        let x = load_bf16_in0(i);
        store_bf16_out(i, x * x);
    }
}

@compute @workgroup_size(1)
fn erf_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, erf_approx(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn ceil_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, ceil(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn floor_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, floor(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn round_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, round(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn sign_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, sign_bf16_val(load_bf16_in0(i)));
    }
}

@compute @workgroup_size(1)
fn affine_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    let mul = bitcast<f32>(kernel_params._pad0);
    let add = bitcast<f32>(kernel_params._pad1);
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, load_bf16_in0(i) * mul + add);
    }
}
