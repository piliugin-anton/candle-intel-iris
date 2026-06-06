enable f16;
// RMS normalization (f16): out = x / sqrt(mean(x^2) + eps) * alpha
//
// Entry point: rms_norm_f16

struct RmsNormParams {
    n_rows: u32,
    n_cols: u32,
    eps_bits: u32,
    _pad: array<u32, 69>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> rms_params: RmsNormParams;

@compute @workgroup_size(32)
fn rms_norm_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= rms_params.n_rows) {
        return;
    }
    let n_cols = rms_params.n_cols;
    let eps = bitcast<f32>(rms_params.eps_bits);
    let row_base = row * n_cols;

    var sum2 = 0.0;
    for (var c = 0u; c < n_cols; c = c + 1u) {
        let v = f32(input0_buf[row_base + c]);
        sum2 += v * v;
    }
    let denom = sqrt(sum2 / f32(n_cols) + eps);

    for (var c = 0u; c < n_cols; c = c + 1u) {
        let idx = row_base + c;
        let scaled = f32(input0_buf[idx]) / denom * f32(input1_buf[c]);
        output_buf[idx] = f16(scaled);
    }
}
