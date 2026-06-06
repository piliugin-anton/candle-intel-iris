// Element-wise binary kernels (u32).

@compute @workgroup_size(WG_SIZE)
fn add_u32(
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
fn sub_u32(
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
fn mul_u32(
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
fn min_u32(
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
fn max_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, max(load_in0(i), load_in1(i)));
    }
}
