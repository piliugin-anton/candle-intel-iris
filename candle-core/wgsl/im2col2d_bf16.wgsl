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

fn write_packed_bf16(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = f32_to_bf16_bits(value);
    let mask = ~(0xFFFFu << shift);
    let contribution = bf16 << shift;
    var old = atomicLoad(&output_buf[word]);
    loop {
        let new_val = (old & mask) | contribution;
        let exch = atomicCompareExchangeWeak(&output_buf[word], old, new_val);
        if (exch.exchanged) {
            break;
        }
        old = exch.old_value;
    }
}
const MAX_DIMS: u32 = 8u;

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
var<storage, read_write> output_buf: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: Im2col2dParams;

@compute @workgroup_size(32)
fn im2col2d_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let h_out = p.h_out;
    let w_out = p.w_out;
    let h_k = p.h_k;
    let w_k = p.w_k;
    let stride = p.stride;
    let padding = p.padding;
    let dilation = p.dilation;
    let src_layout = p.src_layout;
    let h_in = src_layout.dims[2];
    let w_in = src_layout.dims[3];
    let c_in = src_layout.dims[1];

    let dst_s4 = w_k;
    let dst_s3 = h_k * dst_s4;
    let dst_s2 = c_in * dst_s3;
    let dst_s1 = w_out * dst_s2;
    let dst_s0 = h_out * dst_s1;

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
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
            write_packed_bf16(tid, 0.0);
        } else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
            write_packed_bf16(tid, 0.0);
        } else {
            src_h_idx -= padding;
            src_w_idx -= padding;
            let src_i = b_idx * src_layout.strides[0]
                + c_idx * src_layout.strides[1]
                + src_h_idx * src_layout.strides[2]
                + src_w_idx * src_layout.strides[3];
            write_packed_bf16(tid, read_packed_bf16(input_buf[src_i / 2u], src_i));
        }
    }
}
