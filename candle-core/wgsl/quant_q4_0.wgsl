// Quantize f32 activations to Q4_0 blocks (32 elements per block).

const BLOCK_Q4_0_BYTES: u32 = 18u;

@compute @workgroup_size(1)
fn quant_q4_0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    let base = block_idx * 32u;
    let elem_count = quant_params.elem_count;
    if (base >= elem_count) {
        return;
    }

    var amax = 0.0;
    var max_v = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        if (base + j >= elem_count) {
            break;
        }
        let x = quant_src[base + j];
        let ax = abs(x);
        if (ax > amax) {
            amax = ax;
            max_v = x;
        }
    }

    let d = max_v / -8.0;
    let id = select(0.0, 1.0 / d, d != 0.0);
    let d_bytes = f32_to_f16_bytes(d);
    let block_byte = block_idx * BLOCK_Q4_0_BYTES;
    quant_write_byte(block_byte, d_bytes.x);
    quant_write_byte(block_byte + 1u, d_bytes.y);

    for (var j = 0u; j < 16u; j = j + 1u) {
        let x0 = select(0.0, quant_src[base + j] * id, base + j < elem_count);
        let x1 = select(0.0, quant_src[base + j + 16u] * id, base + j + 16u < elem_count);
        let xi0 = min(15u, u32(x0 + 8.5));
        let xi1 = min(15u, u32(x1 + 8.5));
        quant_write_byte(block_byte + 2u + j, xi0 | (xi1 << 4u));
    }
}
