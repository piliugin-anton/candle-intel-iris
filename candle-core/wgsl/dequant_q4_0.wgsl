// Dequantize Q4_0 blocks to f32 (32 elements per block, 18 bytes).

const BLOCK_Q4_0_BYTES: u32 = 18u;

fn q4_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q4_0_BYTES;
}

fn q4_block_d(block_idx: u32) -> f32 {
    let base = q4_block_base(block_idx);
    return f16_bytes_to_f32(dequant_read_byte(base), dequant_read_byte(base + 1u));
}

fn q4_block_qs_byte(block_idx: u32, byte_idx: u32) -> u32 {
    return dequant_read_byte(q4_block_base(block_idx) + 2u + byte_idx);
}

@compute @workgroup_size(256)
fn dequant_q4_0_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let elem_count = dequant_params.elem_count;
    let stride = 256u * num_wg.x;
    for (var i = gid.x; i < elem_count; i = i + stride) {
        let block_idx = i / 32u;
        let elem_in_block = i % 32u;
        let d = q4_block_d(block_idx);
        let j = elem_in_block % 16u;
        let qs_byte = q4_block_qs_byte(block_idx, j);
        let nibble = select(qs_byte & 0x0Fu, qs_byte >> 4u, elem_in_block >= 16u);
        dequant_out[i] = f32(i32(nibble) - 8) * d;
    }
}
