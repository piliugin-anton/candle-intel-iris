fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn read_packed_bf16(packed: u32, elem_idx: u32) -> f32 {
    let byte_off = (elem_idx % 2u) * 2u;
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

const MAX_DIMS: u32 = 8u;

const WG_SIZE: u32 = 32u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct Im2col2dParams {
    dst_numel: u32,
    h_out: u32,
    w_out: u32,
    h_k: u32,
    w_k: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    src_layout: TensorLayout,
    _pad: array<u32, 44>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: Im2col2dParams;

fn is_nchw_contiguous(tensor_layout: TensorLayout) -> bool {
    if (tensor_layout.num_dims != 4u) {
        return false;
    }
    let w = tensor_layout.dims[3];
    let h = tensor_layout.dims[2];
    let c = tensor_layout.dims[1];
    return tensor_layout.strides[3] == 1u
        && tensor_layout.strides[2] == w
        && tensor_layout.strides[1] == h * w
        && tensor_layout.strides[0] == c * h * w;
}

fn im2col_value_at(
    tid: u32,
    p: Im2col2dParams,
    src_layout: TensorLayout,
    nchw: bool,
    plane: u32,
    row: u32,
) -> f32 {
    let h_out = p.h_out;
    let w_out = p.w_out;
    let h_k = p.h_k;
    let w_k = p.w_k;
    let stride = p.stride;
    let padding = p.padding;
    let dilation = p.dilation;
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];

    let dst_s4 = w_k;
    let dst_s3 = h_k * dst_s4;
    let dst_s2 = src_layout.dims[1] * dst_s3;
    let dst_s1 = w_out * dst_s2;
    let dst_s0 = h_out * dst_s1;

    var tmp = tid;
    let b_idx = tmp / dst_s0;
    tmp -= b_idx * dst_s0;
    let h_idx = tmp / dst_s1;
    tmp -= h_idx * dst_s1;
    let w_idx = tmp / dst_s2;
    tmp -= w_idx * dst_s2;
    let c_idx = tmp / dst_s3;
    tmp -= c_idx * dst_s3;
    let h_k_idx = tmp / dst_s4;
    tmp -= h_k_idx * dst_s4;
    let w_k_idx = tmp;

    var src_h_idx = h_idx * stride + h_k_idx * dilation;
    var src_w_idx = w_idx * stride + w_k_idx * dilation;
    if (src_h_idx < padding || src_h_idx >= h_in + padding) {
        return 0.0;
    }
    if (src_w_idx < padding || src_w_idx >= w_in + padding) {
        return 0.0;
    }
    src_h_idx -= padding;
    src_w_idx -= padding;
    var src_i = 0u;
    if (nchw) {
        src_i = src_layout.offset
            + b_idx * plane
            + c_idx * row
            + src_h_idx * w_in
            + src_w_idx;
    } else {
        src_i = src_layout.offset
            + b_idx * src_layout.strides[0]
            + c_idx * src_layout.strides[1]
            + src_h_idx * src_layout.strides[2]
            + src_w_idx * src_layout.strides[3];
    }
    return read_packed_bf16(input_buf[src_i / 2u], src_i);
}

@compute @workgroup_size(WG_SIZE)
fn im2col2d_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = WG_SIZE * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];
    let c_in = src_layout.dims[1];
    let nchw = is_nchw_contiguous(src_layout);
    let plane = c_in * h_in * w_in;
    let row = h_in * w_in;

    // Pair-packed writes: one thread owns each u32 word (no atomic RMW).
    let num_words = (p.dst_numel + 1u) / 2u;
    for (var word = gid.x; word < num_words; word = word + stride_wg) {
        let elem0 = word * 2u;
        let bf0 = f32_to_bf16_bits(im2col_value_at(elem0, p, src_layout, nchw, plane, row));
        var packed = bf0;
        if (elem0 + 1u < p.dst_numel) {
            let bf1 = f32_to_bf16_bits(
                im2col_value_at(elem0 + 1u, p, src_layout, nchw, plane, row),
            );
            packed = bf0 | (bf1 << 16u);
        }
        output_buf[word] = packed;
    }
}
