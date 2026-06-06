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
var<storage, read_write> output_buf: array<u32>;

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

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn load_bf16_in0(idx: u32) -> f32 {
    let elem = buffer_index(idx, kernel_params.in0_layout);
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
    let packed = input0_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn load_bf16_in1(idx: u32) -> f32 {
    let elem = buffer_index(idx, kernel_params.in1_layout);
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
    var packed = output_buf[word];
    packed = (packed & mask) | (bf16 << shift);
    output_buf[word] = packed;
}

fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}
