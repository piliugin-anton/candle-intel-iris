// copy_strided_src: copy a potentially strided source tensor into a contiguous destination.
//
// `input0_buf` holds the source; `output_buf` receives contiguous elements.
// `params.in0_layout` describes the source strides; `params.out_layout` is the
// destination (typically contiguous). Matches CUDA `ucopy_*` kernels.
//
// Entry point: copy_strided_f32

@compute @workgroup_size(WG_SIZE)
fn copy_strided_f32(
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
