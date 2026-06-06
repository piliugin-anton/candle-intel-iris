// Layer normalization: out = (x - mean) / sqrt(var + eps) * alpha + beta
//
// Entry point: layer_norm_f32

struct LayerNormParams {
    n_rows: u32,
    n_cols: u32,
    eps_bits: u32,
    _pad: array<u32, 69>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> input2_buf: array<f32>;

@group(0) @binding(4)
var<storage, read> layer_norm_params: LayerNormParams;

@compute @workgroup_size(32)
fn layer_norm_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        let v = input0_buf[row_base + c];
        sum += v;
        sum2 += v * v;
    }
    let mean = sum / f32(n_cols);
    let variance = sum2 / f32(n_cols) - mean * mean;
    let inv_std = inverseSqrt(variance + eps);

    for (var c = 0u; c < n_cols; c = c + 1u) {
        let idx = row_base + c;
        let normed = (input0_buf[idx] - mean) * inv_std;
        output_buf[idx] = normed * input1_buf[c] + input2_buf[c];
    }
}
