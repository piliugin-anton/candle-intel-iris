// Shared quantized matmul bindings and activation quantization helpers (f16 activations).
//
// Computes dst = lhs @ rhs^T with f16 activations and f32 output.

enable f16;

const QK32: u32 = 32u;

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
var<storage, read> qmm_lhs: array<f16>;

@group(0) @binding(2)
var<storage, read> qmm_rhs: array<u32>;

@group(0) @binding(3)
var<storage, read> qmm_params: QMatMulParams;

fn qmm_read_byte(byte_idx: u32) -> u32 {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (qmm_rhs[word] >> shift) & 0xFFu;
}

fn f16_bytes_to_f32(lo: u32, hi: u32) -> f32 {
    return f32(unpack2x16float(lo | (hi << 8u)).x);
}

fn qmm_lhs_f32(idx: u32) -> f32 {
    return f32(qmm_lhs[idx]);
}

// Returns (q8_d, q8_id) for a 32-element activation slice at lhs_base.
fn q8_0_quant_params(lhs_base: u32) -> vec2<f32> {
    var amax = 0.0;
    for (var j = 0u; j < QK32; j = j + 1u) {
        amax = max(amax, abs(qmm_lhs_f32(lhs_base + j)));
    }
    let q8_d = amax / 127.0;
    let q8_id = select(0.0, 1.0 / q8_d, q8_d != 0.0);
    return vec2<f32>(q8_d, q8_id);
}

fn q8_0_quant_value(v: f32, q8_id: f32) -> i32 {
    return i32(round(v * q8_id));
}
