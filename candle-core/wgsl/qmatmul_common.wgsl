// Quantized matrix multiply (MMVQ) shared definitions.
//
// Computes dst = lhs @ rhs^T where rhs is stored in GGML Q4_0 blocks (transposed
// weight matrix shape n x k). Activations are f32; output is f32.

const QK4_0: u32 = 32u;
const BLOCK_Q4_0_BYTES: u32 = 18u;

struct QMatMulParams {
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    _pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> qmm_out: array<f32>;

@group(0) @binding(1)
var<storage, read> qmm_lhs: array<f32>;

@group(0) @binding(2)
var<storage, read> qmm_rhs: array<u32>;

@group(0) @binding(3)
var<storage, read> qmm_params: QMatMulParams;

fn qmm_read_byte(byte_idx: u32) -> u32 {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (qmm_rhs[word] >> shift) & 0xFFu;
}

fn qmm_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q4_0_BYTES;
}

fn qmm_block_d(block_idx: u32) -> f32 {
    let base = qmm_block_base(block_idx);
    let lo = qmm_read_byte(base);
    let hi = qmm_read_byte(base + 1u);
    return f32(unpack2x16float(lo | (hi << 8u)).x);
}

fn qmm_block_qs_byte(block_idx: u32, byte_idx: u32) -> u32 {
    return qmm_read_byte(qmm_block_base(block_idx) + 2u + byte_idx);
}
