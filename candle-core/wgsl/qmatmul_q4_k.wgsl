// Q4_K quantized matrix multiply: dst = lhs @ rhs^T (f32 activations).
//
// Entry point: qmatmul_q4_k_f32

@compute @workgroup_size(8, 8)
fn qmatmul_q4_k_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = qmm_params.m;
    let n = qmm_params.n;
    let k_dim = qmm_params.k;

    let row = gid.y;
    let col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    let blocks = k_dim / QK_K;
    let lhs_row_base = row * k_dim;
    let rhs_col_base = col * blocks;

    var acc = 0.0;
    for (var b = 0u; b < blocks; b = b + 1u) {
        let lhs_base = lhs_row_base + b * QK_K;
        let block_idx = rhs_col_base + b;
        acc += vec_dot_q4k_q8k(block_idx, lhs_base);
    }

    qmm_out[row * n + col] = acc;
}
