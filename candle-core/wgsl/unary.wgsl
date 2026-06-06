// Element-wise unary kernels (f32).
//
// Entry points: neg_f32, exp_f32, log_f32, sqrt_f32, abs_f32, relu_f32,
//               recip_f32, silu_f32, sigmoid_f32, gelu_f32, gelu_erf_f32,
//               sin_f32, cos_f32, tanh_f32, sqr_f32, erf_f32, ceil_f32,
//               floor_f32, round_f32, sign_f32, affine_f32

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

@compute @workgroup_size(1)
fn neg_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, -load_in0(i));
}

@compute @workgroup_size(1)
fn exp_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, exp(load_in0(i)));
}

@compute @workgroup_size(1)
fn log_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, log(load_in0(i)));
}

@compute @workgroup_size(1)
fn sqrt_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, sqrt(load_in0(i)));
}

@compute @workgroup_size(1)
fn abs_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, abs(load_in0(i)));
}

@compute @workgroup_size(1)
fn relu_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let x = load_in0(i);
    store_out(i, max(x, 0.0));
}

@compute @workgroup_size(1)
fn recip_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, 1.0 / load_in0(i));
}

@compute @workgroup_size(1)
fn silu_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let x = load_in0(i);
    store_out(i, x / (1.0 + exp(-x)));
}

@compute @workgroup_size(1)
fn sigmoid_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let x = load_in0(i);
    store_out(i, 1.0 / (1.0 + exp(-x)));
}

// Tanh approximation (matches Candle `Gelu` / PyTorch `approximate='tanh'`).
const GELU_SQRT_2_OVER_PI: f32 = 0.7978845608028654;

fn gelu_approx(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(GELU_SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

@compute @workgroup_size(1)
fn gelu_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, gelu_approx(load_in0(i)));
}

@compute @workgroup_size(1)
fn gelu_erf_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let x = load_in0(i);
    store_out(i, (erf_approx(x * INV_SQRT_2) + 1.0) * 0.5 * x);
}

@compute @workgroup_size(1)
fn sin_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, sin(load_in0(i)));
}

@compute @workgroup_size(1)
fn cos_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, cos(load_in0(i)));
}

@compute @workgroup_size(1)
fn tanh_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, tanh(load_in0(i)));
}

@compute @workgroup_size(1)
fn sqr_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let x = load_in0(i);
    store_out(i, x * x);
}

@compute @workgroup_size(1)
fn erf_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, erf_approx(load_in0(i)));
}

@compute @workgroup_size(1)
fn ceil_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, ceil(load_in0(i)));
}

@compute @workgroup_size(1)
fn floor_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, floor(load_in0(i)));
}

@compute @workgroup_size(1)
fn round_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, round(load_in0(i)));
}

@compute @workgroup_size(1)
fn sign_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, sign_f32_val(load_in0(i)));
}

// `_pad0` / `_pad1` hold `mul` / `add` as f32 bit patterns (see `KernelUniforms::new_affine`).
@compute @workgroup_size(1)
fn affine_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    let mul = bitcast<f32>(kernel_params._pad0);
    let add = bitcast<f32>(kernel_params._pad1);
    store_out(i, load_in0(i) * mul + add);
}
