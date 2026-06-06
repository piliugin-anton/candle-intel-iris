// Q4_0 quantized matrix multiply: dst = lhs @ rhs^T (f32 activations).
//
// One thread computes one output element. Each K-block quantizes the activation
// slice to Q8_0, then dots with the corresponding Q4_0 weight block.
//
// Entry point: qmatmul_q4_0_f32

const QK: u32 = 32u;

@compute @workgroup_size(8, 8)
fn qmatmul_q4_0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
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

        var amax = 0.0;
        for (var j = 0u; j < QK; j = j + 1u) {
            amax = max(amax, abs(qmm_lhs[lhs_base + j]));
        }
        let q8_d = amax / 127.0;
        let q8_id = select(0.0, 1.0 / q8_d, q8_d != 0.0);

        let block_idx = rhs_col_base + b;
        let d4 = qmm_block_d(block_idx);
        var sum_i = 0;
        for (var j = 0u; j < QK / 2u; j = j + 1u) {
            let qs = qmm_block_qs_byte(block_idx, j);
            let v0 = i32(qs & 0x0Fu) - 8;
            let v1 = i32(qs >> 4u) - 8;
            let y0 = i32(round(qmm_lhs[lhs_base + j] * q8_id));
            let y1 = i32(round(qmm_lhs[lhs_base + QK / 2u + j] * q8_id));
            sum_i += v0 * y0 + v1 * y1;
        }
        acc += f32(sum_i) * d4 * q8_d;
    }

    qmm_out[row * n + col] = acc;
}
