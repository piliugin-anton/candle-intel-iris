// Dequantize Q4_K blocks to f32 (256 elements per block, 144 bytes).

const QK_K: u32 = 256u;
const BLOCK_Q4_K_BYTES: u32 = 144u;

fn q4k_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q4_K_BYTES;
}

fn q4k_block_d(block_idx: u32) -> f32 {
    let base = q4k_block_base(block_idx);
    return f16_bytes_to_f32(dequant_read_byte(base), dequant_read_byte(base + 1u));
}

fn q4k_block_dmin(block_idx: u32) -> f32 {
    let base = q4k_block_base(block_idx);
    return f16_bytes_to_f32(dequant_read_byte(base + 2u), dequant_read_byte(base + 3u));
}

fn q4k_scale_byte(block_idx: u32, j: u32) -> u32 {
    return dequant_read_byte(q4k_block_base(block_idx) + 4u + j);
}

fn q4k_qs_byte(block_idx: u32, i: u32) -> u32 {
    return dequant_read_byte(q4k_block_base(block_idx) + 4u + 12u + i);
}

fn get_scale_min_k4(j: u32, block_idx: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(q4k_scale_byte(block_idx, j) & 63u, q4k_scale_byte(block_idx, j + 4u) & 63u);
    }
    let d = (q4k_scale_byte(block_idx, j + 4u) & 0x0Fu)
        | ((q4k_scale_byte(block_idx, j - 4u) >> 6u) << 4u);
    let m = (q4k_scale_byte(block_idx, j + 4u) >> 4u)
        | ((q4k_scale_byte(block_idx, j) >> 6u) << 4u);
    return vec2<u32>(d, m);
}

@compute @workgroup_size(32)
fn dequant_q4_k_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let block_idx = wg_id.x;
    let tid = local_id.x;
    let il = tid / 8u;
    let ir = tid % 8u;
    let is = 2u * il;
    let n = 4u;

    let out_base = block_idx * QK_K + 64u * il + n * ir;
    let dall = q4k_block_d(block_idx);
    let dmin = q4k_block_dmin(block_idx);
    let q_base = 32u * il + n * ir;

    let sc0 = get_scale_min_k4(is + 0u, block_idx);
    let d1 = dall * f32(sc0.x);
    let m1 = dmin * f32(sc0.y);
    let sc1 = get_scale_min_k4(is + 1u, block_idx);
    let d2 = dall * f32(sc1.x);
    let m2 = dmin * f32(sc1.y);

    for (var l = 0u; l < n; l = l + 1u) {
        let qv = q4k_qs_byte(block_idx, q_base + l);
        dequant_out[out_base + l] = d1 * f32(qv & 0x0Fu) - m1;
        dequant_out[out_base + l + 32u] = d2 * f32(qv >> 4u) - m2;
    }
}
