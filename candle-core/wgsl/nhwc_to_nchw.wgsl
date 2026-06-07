// Permute a contiguous BHWC tensor into contiguous BCHW (NCHW).
// Used after conv2d matmul, which writes (B, H*W, C) in BHWC order.
//
// Entry point: nhwc_to_nchw_f32

const WG_SIZE: u32 = 32u;

struct NhwcToNchwParams {
    elem_count: u32,
    b: u32,
    h: u32,
    w: u32,
    c: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: array<u32, 65>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> params: NhwcToNchwParams;

@compute @workgroup_size(WG_SIZE)
fn nhwc_to_nchw_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = WG_SIZE * num_wg.x;
    let p = params;
    let w = p.w;
    let hw = p.h * w;
    let chw = p.c * hw;
    let src_base = p.src_offset;
    let dst_base = p.dst_offset;

    for (var dst_i = gid.x; dst_i < p.elem_count; dst_i = dst_i + stride_wg) {
        let b_idx = dst_i / chw;
        let rem = dst_i % chw;
        let c_idx = rem / hw;
        let rem2 = rem % hw;
        let h_idx = rem2 / w;
        let w_idx = rem2 % w;
        let src_i = src_base + b_idx * hw * p.c + h_idx * w * p.c + w_idx * p.c + c_idx;
        output_buf[dst_base + dst_i] = input_buf[src_i];
    }
}
