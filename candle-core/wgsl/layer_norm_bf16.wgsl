// Layer normalization (bf16): out = (x - mean) / sqrt(var + eps) * alpha + beta
//
// Entry point: layer_norm_bf16

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

fn pack_bf16_value(packed: u32, elem_idx: u32, value: f32) -> u32 {
    let byte_off = (elem_idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = f32_to_bf16_bits(value);
    let mask = ~(0xFFFFu << shift);
    return (packed & mask) | (bf16 << shift);
}

struct LayerNormParams {
    n_rows: u32,
    n_cols: u32,
    eps_bits: u32,
    _pad: array<u32, 69>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> input2_buf: array<u32>;

@group(0) @binding(4)
var<storage, read> layer_norm_params: LayerNormParams;

@compute @workgroup_size(32)
fn layer_norm_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= layer_norm_params.n_rows) {
        return;
    }
    let n_cols = layer_norm_params.n_cols;
    let eps = bitcast<f32>(layer_norm_params.eps_bits);
    let row_base = row * n_cols;

    var sum = 0.0;
    var sum2 = 0.0;
    for (var c = 0u; c < n_cols; c = c + 1u) {
        let idx = row_base + c;
        let v = read_packed_bf16(input0_buf[idx / 2u], idx);
        sum += v;
        sum2 += v * v;
    }
    let mean = sum / f32(n_cols);
    let variance = sum2 / f32(n_cols) - mean * mean;
    let inv_std = inverseSqrt(variance + eps);

    for (var c = 0u; c < n_cols; c = c + 1u) {
        let idx = row_base + c;
        let x = read_packed_bf16(input0_buf[idx / 2u], idx);
        let a = read_packed_bf16(input1_buf[c / 2u], c);
        let b = read_packed_bf16(input2_buf[c / 2u], c);
        let normed = (x - mean) * inv_std;
        let scaled = normed * a + b;
        output_buf[idx / 2u] = pack_bf16_value(output_buf[idx / 2u], idx, scaled);
    }
}
