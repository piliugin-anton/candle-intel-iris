// Matrix multiplication: f16 inputs, f32 output (Gen11 FP32-compute policy).
//
// Requires `enable f16` and device SHADER_F16 support.

enable f16;

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
var<storage, read> a_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> b_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> mm_params: MatMulParams;

fn mm_elem_index(tensor_layout: TensorLayout, batch: u32, d1: u32, d2: u32) -> u32 {
    if (tensor_layout.num_dims < 3u) {
        return tensor_layout.offset + d1 * tensor_layout.strides[0] + d2 * tensor_layout.strides[1];
    }
    return tensor_layout.offset
        + batch * tensor_layout.strides[0]
        + d1 * tensor_layout.strides[1]
        + d2 * tensor_layout.strides[2];
}

fn mm_load_a(batch: u32, row: u32, col: u32) -> f16 {
    return a_buf[mm_elem_index(mm_params.a_layout, batch, row, col)];
}

fn mm_load_b(batch: u32, row: u32, col: u32) -> f16 {
    return b_buf[mm_elem_index(mm_params.b_layout, batch, row, col)];
}

fn mm_store_c(batch: u32, row: u32, col: u32, value: f32) {
    c_buf[mm_elem_index(mm_params.c_layout, batch, row, col)] = value;
}
