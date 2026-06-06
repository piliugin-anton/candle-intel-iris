// Naive matrix multiply: one thread computes one output element.
//
// Entry point: matmul_naive_f32

@compute @workgroup_size(MATMUL_WG_SIZE, MATMUL_WG_SIZE)
fn matmul_naive_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = mm_params.m;
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = gid.z;

    let row = gid.y;
    let col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    var acc = 0.0;
    for (var k = 0u; k < k_dim; k = k + 1u) {
        acc += mm_load_a(batch, row, k) * mm_load_b(batch, k, col);
    }
    mm_store_c(batch, row, col, acc);
}
