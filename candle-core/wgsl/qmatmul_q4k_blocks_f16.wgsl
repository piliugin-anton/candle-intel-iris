// Q4_K weight block layout (144 bytes per block, QK=256) — f16 activations.

const QK_K: u32 = 256u;
const K_SCALE_SIZE: u32 = 12u;
const BLOCK_Q4_K_BYTES: u32 = 144u;

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

fn q4k_block_base(block_idx: u32) -> u32 {
    return block_idx * BLOCK_Q4_K_BYTES;
}

fn q4k_block_d(block_idx: u32) -> f32 {
    let base = q4k_block_base(block_idx);
    return f16_bytes_to_f32(qmm_read_byte(base), qmm_read_byte(base + 1u));
}

fn q4k_block_dmin(block_idx: u32) -> f32 {
    let base = q4k_block_base(block_idx);
    return f16_bytes_to_f32(qmm_read_byte(base + 2u), qmm_read_byte(base + 3u));
}

fn q4k_block_qs_byte(block_idx: u32, i: u32) -> u32 {
    return qmm_read_byte(q4k_block_base(block_idx) + 4u + K_SCALE_SIZE + i);
}

fn read_u32_le_at(byte_base: u32) -> u32 {
    return qmm_read_byte(byte_base)
        | (qmm_read_byte(byte_base + 1u) << 8u)
        | (qmm_read_byte(byte_base + 2u) << 16u)
        | (qmm_read_byte(byte_base + 3u) << 24u);
}

fn vec_dot_q4k_q8k(block_idx: u32, lhs_base: u32) -> f32 {
    var q8: array<i32, 256>;
    var bsums: array<i32, 16>;

    var amax = 0.0;
    for (var i = 0u; i < QK_K; i = i + 1u) {
        amax = max(amax, abs(qmm_lhs_f32(lhs_base + i)));
    }
    let yd = amax / 127.0;
    let yid = select(0.0, 1.0 / yd, yd != 0.0);
    for (var i = 0u; i < QK_K; i = i + 1u) {
        q8[i] = i32(round(qmm_lhs_f32(lhs_base + i) * yid));
    }
    for (var j = 0u; j < 16u; j = j + 1u) {
        var s = 0;
        let base = j * 16u;
        for (var l = 0u; l < 16u; l = l + 1u) {
            s += q8[base + l];
        }
        bsums[j] = s;
    }

    var aux8: array<i32, 256>;
    var aux16: array<i32, 8>;
    var aux32: array<i32, 8>;
    var scales: array<u32, 8>;
    var mins: array<u32, 8>;

    var a_off = 0u;
    for (var chunk = 0u; chunk < 4u; chunk = chunk + 1u) {
        for (var l = 0u; l < 32u; l = l + 1u) {
            let q4b = q4k_block_qs_byte(block_idx, chunk * 32u + l);
            aux8[a_off + l] = i32(q4b & 0xFu);
        }
        a_off += 32u;
        for (var l = 0u; l < 32u; l = l + 1u) {
            let q4b = q4k_block_qs_byte(block_idx, chunk * 32u + l);
            aux8[a_off + l] = i32(q4b >> 4u);
        }
        a_off += 32u;
    }

    var utmp: array<u32, 4>;
    utmp[0] = read_u32_le_at(q4k_block_base(block_idx) + 4u);
    utmp[1] = read_u32_le_at(q4k_block_base(block_idx) + 8u);
    utmp[2] = read_u32_le_at(q4k_block_base(block_idx) + 12u);

    utmp[3] = ((utmp[2] >> 4u) & KMASK2) | (((utmp[1] >> 6u) & KMASK3) << 4u);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6u) & KMASK3) << 4u);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    for (var i = 0u; i < 4u; i = i + 1u) {
        scales[i] = (utmp[0] >> (i * 8u)) & 0xFFu;
        scales[i + 4u] = (utmp[1] >> (i * 8u)) & 0xFFu;
        mins[i] = (utmp[2] >> (i * 8u)) & 0xFFu;
        mins[i + 4u] = (utmp[3] >> (i * 8u)) & 0xFFu;
    }

    var sumi = 0;
    for (var j = 0u; j < 16u; j = j + 1u) {
        sumi += bsums[j] * i32(mins[j / 2u]);
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        aux32[i] = 0;
    }

    var q8_off = 0u;
    var a_ptr = 0u;
    for (var si = 0u; si < 8u; si = si + 1u) {
        let scale = i32(scales[si]);
        for (var rep = 0u; rep < 4u; rep = rep + 1u) {
            for (var l = 0u; l < 8u; l = l + 1u) {
                aux16[l] = q8[q8_off + l] * aux8[a_ptr + l];
            }
            for (var l = 0u; l < 8u; l = l + 1u) {
                aux32[l] += scale * aux16[l];
            }
            q8_off += 8u;
            a_ptr += 8u;
        }
    }

    let d = q4k_block_d(block_idx) * yd;
    var sumf = 0.0;
    for (var l = 0u; l < 8u; l = l + 1u) {
        sumf += d * f32(aux32[l]);
    }
    let dmin = q4k_block_dmin(block_idx) * yd;
    return sumf - dmin * f32(sumi);
}
