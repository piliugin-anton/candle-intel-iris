// Element-wise comparison kernels (packed bf16 inputs, u8 output).
//
// Entry points: eq_bf16, ne_bf16, lt_bf16, le_bf16, gt_bf16, ge_bf16

fn store_u8_cmp(idx: u32, value: u32) {
    let byte_idx = buffer_index(idx, kernel_params.out_layout);
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    atomicOr(&output_buf[word], (value & 0xFFu) << shift);
}

@compute @workgroup_size(WG_SIZE)
fn eq_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) == load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn ne_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) != load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn lt_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) < load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn le_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) <= load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn gt_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) > load_bf16_in1(i)));
    }
}

@compute @workgroup_size(WG_SIZE)
fn ge_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        store_u8_cmp(i, select(0u, 1u, load_bf16_in0(i) >= load_bf16_in1(i)));
    }
}
