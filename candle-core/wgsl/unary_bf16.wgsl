// Element-wise unary kernels (packed bf16) for LLM-critical ops.
// Serial execution: two bf16 values share one u32 word (no write races).
//
// Entry points: exp_bf16, gelu_bf16, gelu_erf_bf16, silu_bf16, sigmoid_bf16,
//               tanh_bf16, neg_bf16, sqr_bf16

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

fn gelu_approx(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(GELU_SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
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
fn neg_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, -load_bf16_in0(i));
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
