// copy_strided_src for u32 tensors (index ids, argmax output, etc.).
//
// Entry point: copy_strided_u32

@compute @workgroup_size(WG_SIZE)
fn copy_strided_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let src_layout = kernel_params.in0_layout;
    let dst_layout = kernel_params.out_layout;

    for (var i = gid.x; i < count; i = i + stride) {
        var src_idx: u32;
        if (is_contiguous(src_layout)) {
            src_idx = src_layout.offset + i;
        } else {
            src_idx = get_strided_index(i, src_layout);
        }
        let value = input0_buf[src_idx];
        if (is_contiguous(dst_layout)) {
            output_buf[dst_layout.offset + i] = value;
        } else {
            output_buf[get_strided_index(i, dst_layout)] = value;
        }
    }
}
