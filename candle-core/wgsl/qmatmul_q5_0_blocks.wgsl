// Q5_0 weight block layout helpers (22 bytes per block, QK=32).

const BLOCK_Q5_0_BYTES: u32 = 22u;

fn q5_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q5_0_BYTES;
}

fn q5_block_d(block_idx: u32) -> f32 {
    let base = q5_block_base(block_idx);
    return f16_bytes_to_f32(qmm_read_byte(base), qmm_read_byte(base + 1u));
}

fn q5_block_qh(block_idx: u32) -> u32 {
    let base = q5_block_base(block_idx) + 2u;
    return qmm_read_byte(base)
        | (qmm_read_byte(base + 1u) << 8u)
        | (qmm_read_byte(base + 2u) << 16u)
        | (qmm_read_byte(base + 3u) << 24u);
}

fn q5_block_qs_byte(block_idx: u32, byte_idx: u32) -> u32 {
    return qmm_read_byte(q5_block_base(block_idx) + 6u + byte_idx);
}

fn q5_nibble_value(block_idx: u32, j: u32, high: bool) -> i32 {
    let qs_byte = q5_block_qs_byte(block_idx, j);
    let qh = q5_block_qh(block_idx);
    let xh = select(
        ((qh & (1u << j)) >> j) << 4u,
        (qh & (1u << (j + 16u))) >> (j + 12u),
        high,
    );
    let nibble = select(qs_byte & 0x0Fu, qs_byte >> 4u, high);
    return i32(nibble | xh) - 16;
}
