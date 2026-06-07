// Element-wise binary kernels (packed bf16) with grid-stride loops.
// Per-element atomic RMW handles packed-word writes safely.
//
// Entry points: add_bf16, sub_bf16, mul_bf16, div_bf16, min_bf16, max_bf16

@compute @workgroup_size(WG_SIZE)
fn add_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let bias_add = kernel_params._pad2 == BROADCAST_BIAS_ADD;
    for (var i = gid.x; i < count; i = i + stride) {
        if (bias_add) {
            store_bf16_out(i, fused_bias_add_bf16(i));
        } else {
            store_bf16_out(i, load_bf16_in0(i) + load_bf16_in1(i));
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn sub_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, load_bf16_in0(i) - load_bf16_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn mul_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, load_bf16_in0(i) * load_bf16_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn div_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, load_bf16_in0(i) / load_bf16_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn min_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, min(load_bf16_in0(i), load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn max_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, max(load_bf16_in0(i), load_bf16_in1(i)));
    }
}
