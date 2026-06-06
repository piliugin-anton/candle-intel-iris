// copy_strided_src for packed bf16 tensors.
//
// Entry point: copy_strided_bf16

@compute @workgroup_size(WG_SIZE)
fn copy_strided_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;

    for (var i = gid.x; i < count; i = i + stride) {
        let value = load_bf16_in0(i);
        store_bf16_out(i, value);
    }
}
