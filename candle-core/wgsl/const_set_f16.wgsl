// Fill a (possibly strided) f16 tensor with a scalar.
//
// `_pad0` holds the f16 value as bits in the low 16 bits.
// Entry point: const_set_f16

@compute @workgroup_size(WG_SIZE)
fn const_set_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let value = f16(bitcast<f32>(kernel_params._pad0)); // f32 bits of the scalar value
    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, value);
    }
}
