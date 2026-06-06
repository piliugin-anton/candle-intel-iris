// Fill a (possibly strided) bf16 tensor with a scalar (packed u32 words).
//
// `_pad0` holds bf16 bits in the low 16 bits.
// Entry point: const_set_bf16

@compute @workgroup_size(WG_SIZE)
fn const_set_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let value = bf16_bits_to_f32(kernel_params._pad0 & 0xFFFFu);

    for (var i = gid.x; i < count; i = i + stride) {
        store_bf16_out(i, value);
    }
}
