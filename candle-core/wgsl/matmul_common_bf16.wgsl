// Matrix multiplication shared definitions for packed bf16 (u32 words, f32 math).
//
// Two bf16 values per u32 word; accumulation in f32.

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
var<storage, read_write> c_buf: array<u32>;

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

fn write_bf16_c(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = (bitcast<u32>(value) >> 16u) & 0xFFFFu;
    let mask = ~(0xFFFFu << shift);
    var packed = c_buf[word];
    packed = (packed & mask) | (bf16 << shift);
    c_buf[word] = packed;
}

fn mm_load_a(row: u32, col: u32) -> f32 {
    return read_bf16_a(row * mm_params.k + col);
}

fn mm_load_b(row: u32, col: u32) -> f32 {
    return read_bf16_b(row * mm_params.n + col);
}

fn mm_store_c(row: u32, col: u32, value: f32) {
    write_bf16_c(row * mm_params.n + col, value);
}
