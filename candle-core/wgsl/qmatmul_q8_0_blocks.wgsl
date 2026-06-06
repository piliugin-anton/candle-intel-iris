// Q8_0 weight block layout helpers (34 bytes per block, QK=32).

const BLOCK_Q8_0_BYTES: u32 = 34u;

fn q8_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q8_0_BYTES;
}

fn q8_block_d(block_idx: u32) -> f32 {
    let base = q8_block_base(block_idx);
    return f16_bytes_to_f32(qmm_read_byte(base), qmm_read_byte(base + 1u));
}

fn q8_block_qs(block_idx: u32, j: u32) -> i32 {
    let byte_idx = q8_block_base(block_idx) + 2u + j;
    let b = qmm_read_byte(byte_idx);
    return i32(b) - select(0, 256, b > 127u);
}
