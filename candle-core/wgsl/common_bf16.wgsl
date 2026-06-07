// Shared layout utilities and packed bf16 buffers for element-wise kernels.

const MAX_DIMS: u32 = 8u;
const WG_SIZE: u32 = 32u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct KernelParams {
    elem_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    out_layout: TensorLayout,
    in0_layout: TensorLayout,
    in1_layout: TensorLayout,
    _tail_pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> kernel_params: KernelParams;

fn get_strided_index(idx: u32, tensor_layout: TensorLayout) -> u32 {
    var remaining = idx;
    var strided_i = 0u;
    let num_dims = tensor_layout.num_dims;
    for (var d = 0u; d < num_dims; d = d + 1u) {
        let dim_idx = num_dims - 1u - d;
        let dim = tensor_layout.dims[dim_idx];
        let stride = tensor_layout.strides[dim_idx];
        strided_i += (remaining % dim) * stride;
        remaining /= dim;
    }
    return tensor_layout.offset + strided_i;
}

fn is_contiguous(tensor_layout: TensorLayout) -> bool {
    var acc = 1u;
    let num_dims = tensor_layout.num_dims;
    for (var d = 0u; d < num_dims; d = d + 1u) {
        let dim_idx = num_dims - 1u - d;
        let dim = tensor_layout.dims[dim_idx];
        if (dim > 1u && acc != tensor_layout.strides[dim_idx]) {
            return false;
        }
        acc *= dim;
    }
    return true;
}

fn buffer_index(idx: u32, tensor_layout: TensorLayout) -> u32 {
    if (is_contiguous(tensor_layout)) {
        return tensor_layout.offset + idx;
    }
    return get_strided_index(idx, tensor_layout);
}

const BROADCAST_BIAS_ADD: u32 = 1u;

fn nchw_coords(idx: u32, tensor_layout: TensorLayout) -> vec4<u32> {
    let w = tensor_layout.dims[3];
    let hw = tensor_layout.dims[2] * w;
    let chw = tensor_layout.dims[1] * hw;
    let b_idx = idx / chw;
    let rem = idx % chw;
    let c_idx = rem / hw;
    let rem2 = rem % hw;
    let h_idx = rem2 / w;
    let w_idx = rem2 % w;
    return vec4<u32>(b_idx, c_idx, h_idx, w_idx);
}

fn buffer_index_bias_in0(idx: u32) -> u32 {
    let in0 = kernel_params.in0_layout;
    let coords = nchw_coords(idx, kernel_params.out_layout);
    return in0.offset
        + coords.x * in0.strides[0]
        + coords.z * in0.strides[2]
        + coords.w * in0.strides[3];
}

fn buffer_index_bias_in1(idx: u32) -> u32 {
    let in1 = kernel_params.in1_layout;
    let coords = nchw_coords(idx, kernel_params.out_layout);
    return in1.offset + coords.x * in1.strides[0] + coords.y * in1.strides[1];
}

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn load_bf16_in0(idx: u32) -> f32 {
    var elem = 0u;
    if (kernel_params._pad2 == BROADCAST_BIAS_ADD) {
        elem = buffer_index_bias_in0(idx);
    } else {
        elem = buffer_index(idx, kernel_params.in0_layout);
    }
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
    let packed = input0_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn load_bf16_in1(idx: u32) -> f32 {
    var elem = 0u;
    if (kernel_params._pad2 == BROADCAST_BIAS_ADD) {
        elem = buffer_index_bias_in1(idx);
    } else {
        elem = buffer_index(idx, kernel_params.in1_layout);
    }
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
    let packed = input1_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn store_bf16_out(idx: u32, value: f32) {
    let out_layout = kernel_params.out_layout;
    var elem = out_layout.offset + idx;
    if (!is_contiguous(out_layout)) {
        elem = get_strided_index(idx, out_layout);
    }
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
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

fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}

fn fused_bias_add_bf16(idx: u32) -> f32 {
    let coords = nchw_coords(idx, kernel_params.out_layout);
    let in0 = kernel_params.in0_layout;
    let in1 = kernel_params.in1_layout;
    let i0 = in0.offset + coords.x * in0.strides[0] + coords.z * in0.strides[2] + coords.w;
    let i1 = in1.offset + coords.x * in1.strides[0] + coords.y * in1.strides[1];
    let word0 = i0 / 2u;
    let off0 = (i0 % 2u) * 2u;
    let packed0 = input0_buf[word0];
    let bf0 = (packed0 >> (off0 * 8u)) & 0xFFFFu;
    let word1 = i1 / 2u;
    let off1 = (i1 % 2u) * 2u;
    let packed1 = input1_buf[word1];
    let bf1 = (packed1 >> (off1 * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf0) + bf16_bits_to_f32(bf1);
}
