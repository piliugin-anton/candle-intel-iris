// Element-wise binary kernels (f32).
//
// Entry points: add_f32, sub_f32, mul_f32, div_f32, min_f32, max_f32

@compute @workgroup_size(1)
fn add_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, load_in0(i) + load_in1(i));
}

@compute @workgroup_size(1)
fn sub_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, load_in0(i) - load_in1(i));
}

@compute @workgroup_size(1)
fn mul_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, load_in0(i) * load_in1(i));
}

@compute @workgroup_size(1)
fn div_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, load_in0(i) / load_in1(i));
}

@compute @workgroup_size(1)
fn min_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, min(load_in0(i), load_in1(i)));
}

@compute @workgroup_size(1)
fn max_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let i = wg_id.x;
    if (i >= kernel_params.elem_count) {
        return;
    }
    store_out(i, max(load_in0(i), load_in1(i)));
}
