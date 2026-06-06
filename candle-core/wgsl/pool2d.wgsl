// 2D average and max pooling (f32).
//
// Entry points: avg_pool2d_f32, max_pool2d_f32

const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct Pool2dParams {
    k_h: u32,
    k_w: u32,
    s_h: u32,
    s_w: u32,
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
var<storage, read> params: Pool2dParams;

@compute @workgroup_size(32)
fn avg_pool2d_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let c = src_layout.dims[1];
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];
    let k_h = p.k_h;
    let k_w = p.k_w;
    let s_h = p.s_h;
    let s_w = p.s_w;
    let h_out = (h_in - k_h) / s_h + 1u;
    let w_out = (w_in - k_w) / s_w + 1u;
    let scale = 1.0 / f32(k_h * k_w);

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (h_out * w_out * c);
        let c_idx = (tid / (h_out * w_out)) % c;
        let h_idx = (tid / w_out) % h_out;
        let w_idx = tid % w_out;

        let src_idx0 = src_layout.offset + b_idx * src_layout.strides[0];
        var sum = 0.0;
        for (var kh = 0u; kh < k_h; kh = kh + 1u) {
            let src_h = s_h * h_idx + kh;
            if (src_h >= h_in) {
                continue;
            }
            for (var kw = 0u; kw < k_w; kw = kw + 1u) {
                let src_w = s_w * w_idx + kw;
                if (src_w >= w_in) {
                    continue;
                }
                let src_idx = src_idx0
                    + c_idx * src_layout.strides[1]
                    + src_h * src_layout.strides[2]
                    + src_w * src_layout.strides[3];
                sum += input_buf[src_idx];
            }
        }
        output_buf[tid] = sum * scale;
    }
}

@compute @workgroup_size(32)
fn max_pool2d_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let c = src_layout.dims[1];
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];
    let k_h = p.k_h;
    let k_w = p.k_w;
    let s_h = p.s_h;
    let s_w = p.s_w;
    let h_out = (h_in - k_h) / s_h + 1u;
    let w_out = (w_in - k_w) / s_w + 1u;

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (h_out * w_out * c);
        let c_idx = (tid / (h_out * w_out)) % c;
        let h_idx = (tid / w_out) % h_out;
        let w_idx = tid % w_out;

        let src_idx0 = src_layout.offset + b_idx * src_layout.strides[0];
        var best = 0.0;
        var found = false;
        for (var kh = 0u; kh < k_h; kh = kh + 1u) {
            let src_h = s_h * h_idx + kh;
            if (src_h >= h_in) {
                continue;
            }
            for (var kw = 0u; kw < k_w; kw = kw + 1u) {
                let src_w = s_w * w_idx + kw;
                if (src_w >= w_in) {
                    continue;
                }
                let src_idx = src_idx0
                    + c_idx * src_layout.strides[1]
                    + src_h * src_layout.strides[2]
                    + src_w * src_layout.strides[3];
                let v = input_buf[src_idx];
                if (found) {
                    best = max(best, v);
                } else {
                    best = v;
                    found = true;
                }
            }
        }
        output_buf[tid] = best;
    }
}
