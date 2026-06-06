// Dequantize Q8_0 blocks to f32 (32 elements per block, 34 bytes).

const BLOCK_Q8_0_BYTES: u32 = 34u;

fn q8_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q8_0_BYTES;
}

fn q8_block_d(block_idx: u32) -> f32 {
    let base = q8_block_base(block_idx);
    return f16_bytes_to_f32(dequant_read_byte(base), dequant_read_byte(base + 1u));
}

fn q8_block_qs(block_idx: u32, j: u32) -> i32 {
    let b = dequant_read_byte(q8_block_base(block_idx) + 2u + j);
    return i32(b) - select(0, 256, b > 127u);
}

@compute @workgroup_size(256)
fn dequant_q8_0_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let elem_count = dequant_params.elem_count;
    let stride = 256u * num_wg.x;
    for (var i = gid.x; i < elem_count; i = i + stride) {
        let block_idx = i / 32u;
        let j = i % 32u;
        dequant_out[i] = f32(q8_block_qs(block_idx, j)) * q8_block_d(block_idx);
    }
}
