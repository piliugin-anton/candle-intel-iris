// 1D GEMV-style matrix multiply for thin M or N (one thread per output element).
//
// Entry points: matmul_gemv_f32 (M == 1), matmul_gemv_col_f32 (N == 1)

const WG_SIZE: u32 = 32u;

@compute @workgroup_size(WG_SIZE)
fn matmul_gemv_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = gid.z;
    let col = gid.x;
    if (col >= n) {
        return;
    }

    let row = 0u;
    var acc = 0.0;
    for (var k = 0u; k < k_dim; k = k + 1u) {
        acc = fma(mm_load_a(batch, row, k), mm_load_b(batch, k, col), acc);
    }
    mm_store_c(batch, row, col, acc);
}

@compute @workgroup_size(WG_SIZE)
fn matmul_gemv_col_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = mm_params.m;
    let k_dim = mm_params.k;
    let batch = gid.z;
    let row = gid.x;
    if (row >= m) {
        return;
    }

    let col = 0u;
    var acc = 0.0;
    for (var k = 0u; k < k_dim; k = k + 1u) {
        acc = fma(mm_load_a(batch, row, k), mm_load_b(batch, k, col), acc);
    }
    mm_store_c(batch, row, col, acc);
}
