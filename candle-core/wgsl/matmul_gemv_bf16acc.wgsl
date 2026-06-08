// Packed bf16 GEMV with pair-packed writes (one u32 per two outputs, no CAS loop).

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn gemv_store_bf16_pair(batch: u32, row: u32, col: u32, v0: f32, v1: f32) {
    let elem0 = mm_elem_index(mm_params.c_layout, batch, row, col);
    let word = elem0 / 2u;
    let lo = f32_to_bf16_bits(v0);
    let hi = f32_to_bf16_bits(v1);
    atomicStore(&c_buf[word], lo | (hi << 16u));
}

@compute @workgroup_size(WG_SIZE)
fn matmul_gemv_bf16acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = gid.z;
    let col = gid.x * 2u;
    if (col >= n) {
        return;
    }
    let v0 = gemv_dot_k(batch, 0u, col, k_dim);
    if (col + 1u < n) {
        let v1 = gemv_dot_k(batch, 0u, col + 1u, k_dim);
        gemv_store_bf16_pair(batch, 0u, col, v0, v1);
    } else {
        write_bf16_c(mm_elem_index(mm_params.c_layout, batch, 0u, col), v0);
    }
}

@compute @workgroup_size(WG_SIZE)
fn matmul_gemv_col_bf16acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = mm_params.m;
    let k_dim = mm_params.k;
    let batch = gid.z;
    let row = gid.x * 2u;
    if (row >= m) {
        return;
    }
    let v0 = gemv_dot_k(batch, row, 0u, k_dim);
    if (row + 1u < m) {
        let v1 = gemv_dot_k(batch, row + 1u, 0u, k_dim);
        gemv_store_bf16_pair(batch, row, 0u, v0, v1);
    } else {
        write_bf16_c(mm_elem_index(mm_params.c_layout, batch, row, 0u), v0);
    }
}
