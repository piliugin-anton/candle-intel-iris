// Element-wise unary kernels (f32).
//
// Entry points: neg_f32, exp_f32, log_f32, sqrt_f32, abs_f32, relu_f32,
//               recip_f32, silu_f32, sigmoid_f32

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
