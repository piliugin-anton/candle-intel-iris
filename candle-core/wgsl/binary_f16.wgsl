// Element-wise binary kernels (f16) with grid-stride loops and strided/broadcast layouts.
//
// Entry points: add_f16, sub_f16, mul_f16, div_f16, min_f16, max_f16

@compute @workgroup_size(WG_SIZE)
fn add_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) + load_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn sub_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) - load_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn mul_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) * load_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn div_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, load_in0(i) / load_in1(i));
    }
}

@compute @workgroup_size(WG_SIZE)
fn min_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, min(load_in0(i), load_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn max_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, max(load_in0(i), load_in1(i)));
    }
}
