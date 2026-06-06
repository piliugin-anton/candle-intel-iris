// Bilinear 2D upsampling (f32). Entry point: upsample_bilinear2d_f32

const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct UpsampleBilinear2dParams {
    dst_h: u32,
    dst_w: u32,
    align_corners: u32,
    has_scale_h: u32,
    scale_h_bits: u32,
    has_scale_w: u32,
    scale_w_bits: u32,
    dst_numel: u32,
    src_layout: TensorLayout,
    _pad: array<u32, 42>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> params: UpsampleBilinear2dParams;

@compute @workgroup_size(32)
fn upsample_bilinear2d_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let h_out = p.dst_h;
    let w_out = p.dst_w;
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];
    let c = src_layout.dims[1];
    let align_corners = p.align_corners != 0u;

    var h_scale: f32;
    var w_scale: f32;
    if (align_corners) {
        h_scale = select(0.0, f32(h_in - 1u) / f32(h_out - 1u), h_out > 1u);
        w_scale = select(0.0, f32(w_in - 1u) / f32(w_out - 1u), w_out > 1u);
    } else {
        h_scale = select(f32(h_in) / f32(h_out), 1.0 / bitcast<f32>(p.scale_h_bits), p.has_scale_h != 0u);
        w_scale = select(f32(w_in) / f32(w_out), 1.0 / bitcast<f32>(p.scale_w_bits), p.has_scale_w != 0u);
    }

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (c * h_out * w_out);
        let c_idx = (tid / (h_out * w_out)) % c;
        let dst_h = (tid / w_out) % h_out;
        let dst_w = tid % w_out;

        var src_h_fp: f32;
        var src_w_fp: f32;
        if (align_corners) {
            src_h_fp = h_scale * f32(dst_h);
            src_w_fp = w_scale * f32(dst_w);
        } else {
            src_h_fp = h_scale * (f32(dst_h) + 0.5) - 0.5;
            src_w_fp = w_scale * (f32(dst_w) + 0.5) - 0.5;
        }
        src_h_fp = max(0.0, src_h_fp);
        src_w_fp = max(0.0, src_w_fp);

        let h0 = u32(floor(src_h_fp));
        let w0 = u32(floor(src_w_fp));
        let h1 = min(h0 + 1u, h_in - 1u);
        let w1 = min(w0 + 1u, w_in - 1u);
        let weight_h = clamp(src_h_fp - f32(h0), 0.0, 1.0);
        let weight_w = clamp(src_w_fp - f32(w0), 0.0, 1.0);

        let base = src_layout.offset
            + b_idx * src_layout.strides[0]
            + c_idx * src_layout.strides[1];
        let v00 = input_buf[base + h0 * src_layout.strides[2] + w0 * src_layout.strides[3]];
        let v10 = input_buf[base + h0 * src_layout.strides[2] + w1 * src_layout.strides[3]];
        let v01 = input_buf[base + h1 * src_layout.strides[2] + w0 * src_layout.strides[3]];
        let v11 = input_buf[base + h1 * src_layout.strides[2] + w1 * src_layout.strides[3]];

        let v_top = v00 * (1.0 - weight_w) + v10 * weight_w;
        let v_bottom = v01 * (1.0 - weight_w) + v11 * weight_w;
        output_buf[tid] = v_top * (1.0 - weight_h) + v_bottom * weight_h;
    }
}
