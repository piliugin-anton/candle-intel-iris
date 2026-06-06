// Element-wise binary kernels (f32) with grid-stride loops and strided/broadcast layouts.
//
// Entry points: add_f32, sub_f32, mul_f32, div_f32, min_f32, max_f32

@compute @workgroup_size(WG_SIZE)
fn add_f32(
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
fn sub_f32(
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
fn mul_f32(
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
fn div_f32(
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
fn min_f32(
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
fn max_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, max(load_in0(i), load_in1(i)));
    }
}
