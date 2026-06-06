enable f16;
// 2D region copy for f32 tensors (used by Tensor::cat).
//
// Uniforms (`Copy2dParams`): d1, d2, src_stride, dst_stride, src_offset, dst_offset.
// Entry point: copy2d_f16

struct Copy2dParams {
    d1: u32,
    d2: u32,
    src_stride: u32,
    dst_stride: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: array<u32, 66>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> copy2d_params: Copy2dParams;

@compute @workgroup_size(32)
fn copy2d_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = 32u * num_wg.x;
    let total = copy2d_params.d1 * copy2d_params.d2;
    let src_stride = copy2d_params.src_stride;
    let dst_stride = copy2d_params.dst_stride;
    let src_base = copy2d_params.src_offset;
    let dst_base = copy2d_params.dst_offset;
    let d2 = copy2d_params.d2;

    for (var flat = gid.x; flat < total; flat = flat + stride) {
        let row = flat / d2;
        let col = flat % d2;
        let src_idx = src_base + row * src_stride + col;
        let dst_idx = dst_base + row * dst_stride + col;
        output_buf[dst_idx] = input0_buf[src_idx];
    }
}
