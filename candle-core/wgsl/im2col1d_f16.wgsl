enable f16;
// im2col for 1D convolution (f32). Entry point: im2col1d_f16

const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct Im2col1dParams {
    dst_numel: u32,
    l_out: u32,
    l_k: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    _align: array<u32, 2>,
    src_layout: TensorLayout,
    _pad: array<u32, 44>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> params: Im2col1dParams;

@compute @workgroup_size(32)
fn im2col1d_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let l_out = p.l_out;
    let l_k = p.l_k;
    let stride = p.stride;
    let padding = p.padding;
    let dilation = p.dilation;
    let src_layout = p.src_layout;
    let l_in = src_layout.dims[2];

    let dst_s2 = l_k;
    let dst_s1 = src_layout.dims[1] * dst_s2;
    let dst_s0 = l_out * dst_s1;

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        var tmp = tid;
        let b_idx = tmp / dst_s0;
        tmp -= b_idx * dst_s0;
        let l_idx = tmp / dst_s1;
        tmp -= l_idx * dst_s1;
        let c_idx = tmp / dst_s2;
        tmp -= c_idx * dst_s2;
        let l_k_idx = tmp;

        var src_l_idx = l_idx * stride + l_k_idx * dilation;
        if (src_l_idx < padding || src_l_idx >= l_in + padding) {
            output_buf[tid] = 0.0;
        } else {
            src_l_idx -= padding;
            let src_i = b_idx * src_layout.strides[0]
                + c_idx * src_layout.strides[1]
                + src_l_idx * src_layout.strides[2];
            output_buf[tid] = input_buf[src_i];
        }
    }
}
