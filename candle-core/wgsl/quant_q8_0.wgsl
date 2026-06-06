// Quantize f32 activations to Q8_0 blocks (32 elements per block).

const BLOCK_Q8_0_BYTES: u32 = 34u;

@compute @workgroup_size(1)
fn quant_q8_0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    let base = block_idx * 32u;
    let elem_count = quant_params.elem_count;
    if (base >= elem_count) {
        return;
    }

    var amax = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        if (base + j >= elem_count) {
            break;
        }
        amax = max(amax, abs(quant_src[base + j]));
    }

    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d != 0.0);
    let d_bytes = f32_to_f16_bytes(d);
    let block_byte = block_idx * BLOCK_Q8_0_BYTES;
    quant_write_byte(block_byte, d_bytes.x);
    quant_write_byte(block_byte + 1u, d_bytes.y);

    for (var j = 0u; j < 32u; j = j + 1u) {
        let x = select(0.0, quant_src[base + j], base + j < elem_count);
        let q = i32(round(x * id));
        let byte = u32(q) & 0xFFu;
        quant_write_byte(block_byte + 2u + j, byte);
    }
}
