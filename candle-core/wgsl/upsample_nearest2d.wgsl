// Nearest-neighbor 2D upsampling (f32). Entry point: upsample_nearest2d_f32

const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct UpsampleNearest2dParams {
    dst_h: u32,
    dst_w: u32,
    scale_h_bits: u32,
    scale_w_bits: u32,
    dst_numel: u32,
    _align: array<u32, 3>,
    src_layout: TensorLayout,
    _pad: array<u32, 44>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> params: UpsampleNearest2dParams;

@compute @workgroup_size(32)
fn upsample_nearest2d_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let dst_h = p.dst_h;
    let dst_w = p.dst_w;
    let src_h = src_layout.dims[2];
    let src_w = src_layout.dims[3];
    let scale_h = bitcast<f32>(p.scale_h_bits);
    let scale_w = bitcast<f32>(p.scale_w_bits);
    let c = src_layout.dims[1];

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (c * dst_h * dst_w);
        let c_idx = (tid / (dst_h * dst_w)) % c;
        let dst_h_idx = (tid / dst_w) % dst_h;
        let dst_w_idx = tid % dst_w;
        let src_h_idx = min(src_h - 1u, u32(f32(dst_h_idx) * scale_h));
        let src_w_idx = min(src_w - 1u, u32(f32(dst_w_idx) * scale_w));
        let src_i = src_layout.offset
            + b_idx * src_layout.strides[0]
            + c_idx * src_layout.strides[1]
            + src_h_idx * src_layout.strides[2]
            + src_w_idx * src_layout.strides[3];
        output_buf[tid] = input_buf[src_i];
    }
}
