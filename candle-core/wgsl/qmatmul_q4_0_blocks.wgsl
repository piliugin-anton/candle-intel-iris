// Q4_0 weight block layout helpers (18 bytes per block, QK=32).

const BLOCK_Q4_0_BYTES: u32 = 18u;

fn q4_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q4_0_BYTES;
}

fn q4_block_d(block_idx: u32) -> f32 {
    let base = q4_block_base(block_idx);
    return f16_bytes_to_f32(qmm_read_byte(base), qmm_read_byte(base + 1u));
}

fn q4_block_qs_byte(block_idx: u32, byte_idx: u32) -> u32 {
    return qmm_read_byte(q4_block_base(block_idx) + 2u + byte_idx);
}
