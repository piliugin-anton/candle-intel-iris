// Fill a (possibly strided) f32 tensor with a scalar.
//
// `_pad0` holds the f32 value as bits (`KernelUniforms::new_const_set`).
// Entry point: const_set_f32

@compute @workgroup_size(WG_SIZE)
fn const_set_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let value = bitcast<f32>(kernel_params._pad0);
    let out_layout = kernel_params.out_layout;

    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, value);
    }
}
