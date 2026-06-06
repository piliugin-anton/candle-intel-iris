// Matrix multiplication shared definitions.
//
// Computes C = A @ B for contiguous row-major matrices. Batch, M, N, K are
// passed explicitly in the uniform header; buffer bindings include any slice offset.

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
var<storage, read> a_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> b_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> mm_params: MatMulParams;

fn mm_load_a(row: u32, col: u32) -> f32 {
    return a_buf[row * mm_params.k + col];
}

fn mm_load_b(row: u32, col: u32) -> f32 {
    return b_buf[row * mm_params.n + col];
}

fn mm_store_c(row: u32, col: u32, value: f32) {
    c_buf[row * mm_params.n + col] = value;
}
