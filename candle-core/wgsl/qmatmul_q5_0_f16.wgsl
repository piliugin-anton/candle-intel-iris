// Q5_0 quantized matrix multiply: dst = lhs @ rhs^T (f16 activations).
//
// Entry point: qmatmul_q5_0_f16

const QK: u32 = 32u;

@compute @workgroup_size(8, 8)
fn qmatmul_q5_0_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = qmm_params.m;
    let n = qmm_params.n;
    let k_dim = qmm_params.k;

    let row = gid.y;
    let col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    let blocks = k_dim / QK;
    let lhs_row_base = row * k_dim;
    let rhs_col_base = col * blocks;

    var acc = 0.0;
    for (var b = 0u; b < blocks; b = b + 1u) {
        let lhs_base = lhs_row_base + b * QK;
        let qparams = q8_0_quant_params(lhs_base);
        let q8_d = qparams.x;
        let q8_id = qparams.y;

        let block_idx = rhs_col_base + b;
        let d5 = q5_block_d(block_idx);
        var sum_i = 0;
        for (var j = 0u; j < QK / 2u; j = j + 1u) {
            let v0 = q5_nibble_value(block_idx, j, false);
            let v1 = q5_nibble_value(block_idx, j, true);
            let y0 = q8_0_quant_value(qmm_lhs_f32(lhs_base + j), q8_id);
            let y1 = q8_0_quant_value(qmm_lhs_f32(lhs_base + QK / 2u + j), q8_id);
            sum_i += v0 * y0 + v1 * y1;
        }
        acc += f32(sum_i) * d5 * q8_d;
    }

    qmm_out[row * n + col] = acc;
}
