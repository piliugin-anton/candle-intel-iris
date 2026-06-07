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

    let both_contiguous = is_contiguous(src_layout) && is_contiguous(dst_layout);
    if (both_contiguous && count >= 4u) {
        let src_base = src_layout.offset;
        let dst_base = dst_layout.offset;
        let vec_stride = stride * 4u;
        let vec_end = (count / 4u) * 4u;
        for (var i = gid.x * 4u; i < vec_end; i = i + vec_stride) {
            let v = vec4<f32>(
                input0_buf[src_base + i],
                input0_buf[src_base + i + 1u],
                input0_buf[src_base + i + 2u],
                input0_buf[src_base + i + 3u],
            );
            output_buf[dst_base + i] = v.x;
            output_buf[dst_base + i + 1u] = v.y;
            output_buf[dst_base + i + 2u] = v.z;
            output_buf[dst_base + i + 3u] = v.w;
        }
        // Tail: one thread per remaining element (not max(vec_end, gid.x), which
        // would make every thread rewrite the same tail indices).
        for (var j = vec_end + gid.x; j < count; j = j + stride) {
            output_buf[dst_base + j] = input0_buf[src_base + j];
        }
        return;
    }
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
