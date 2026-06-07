// 1D GEMV-style matrix multiply for thin M or N (one thread per output element).
//
// Entry points: matmul_gemv_f32 (M == 1), matmul_gemv_col_f32 (N == 1)

const WG_SIZE: u32 = 32u;
const VEC: u32 = 4u;

fn gemv_dot_k(batch: u32, row: u32, col: u32, k_dim: u32) -> f32 {
    var acc = 0.0;
    let k_vec = (k_dim / VEC) * VEC;
    for (var k = 0u; k < k_vec; k = k + VEC) {
        let a_vec = vec4<f32>(
            mm_load_a(batch, row, k),
            mm_load_a(batch, row, k + 1u),
            mm_load_a(batch, row, k + 2u),
            mm_load_a(batch, row, k + 3u),
        );
        let b_vec = vec4<f32>(
            mm_load_b(batch, k, col),
            mm_load_b(batch, k + 1u, col),
            mm_load_b(batch, k + 2u, col),
            mm_load_b(batch, k + 3u, col),
        );
        acc += dot(a_vec, b_vec);
    }
    for (var k = k_vec; k < k_dim; k = k + 1u) {
        acc = fma(mm_load_a(batch, row, k), mm_load_b(batch, k, col), acc);
    }
    return acc;
}

@compute @workgroup_size(WG_SIZE)
fn matmul_gemv_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = gid.z;
    let col = gid.x;
    if (col >= n) {
        return;
    }
    mm_store_c(batch, 0u, col, gemv_dot_k(batch, 0u, col, k_dim));
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
    mm_store_c(batch, row, 0u, gemv_dot_k(batch, row, 0u, k_dim));
}
