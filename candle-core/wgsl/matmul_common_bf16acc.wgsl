// Matrix multiplication: packed bf16 inputs (u32 words), f32 output.
//
// Avoids full-tensor bf16→f32 upcast on Gen11 (FP32 compute policy).

const MAX_DIMS: u32 = 8u;
const MATMUL_WG_SIZE: u32 = 16u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct MatMulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    _tail_pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> c_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> a_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> b_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> mm_params: MatMulParams;

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn read_bf16_a(elem_idx: u32) -> f32 {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let packed = a_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn read_bf16_b(elem_idx: u32) -> f32 {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let packed = b_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn mm_elem_index(tensor_layout: TensorLayout, batch: u32, d1: u32, d2: u32) -> u32 {
    if (tensor_layout.num_dims < 3u) {
        return d1 * tensor_layout.strides[0] + d2 * tensor_layout.strides[1];
    }
    return batch * tensor_layout.strides[0] + d1 * tensor_layout.strides[1] + d2 * tensor_layout.strides[2];
}

fn mm_load_a(batch: u32, row: u32, col: u32) -> f32 {
    return read_bf16_a(mm_elem_index(mm_params.a_layout, batch, row, col));
}

fn mm_load_b(batch: u32, row: u32, col: u32) -> f32 {
    return read_bf16_b(mm_elem_index(mm_params.b_layout, batch, row, col));
}

fn mm_store_c(batch: u32, row: u32, col: u32, value: f32) {
    c_buf[mm_elem_index(mm_params.c_layout, batch, row, col)] = value;
}
