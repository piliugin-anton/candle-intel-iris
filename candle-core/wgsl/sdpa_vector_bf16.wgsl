// Fused scaled dot-product attention (decode / vector path, bf16).
//
// Entry point: sdpa_vector_bf16

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn read_packed_bf16(packed: u32, elem_idx: u32) -> f32 {
    let byte_off = (elem_idx % 2u) * 2u;
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn pack_bf16_value(packed: u32, elem_idx: u32, value: f32) -> u32 {
    let byte_off = (elem_idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = f32_to_bf16_bits(value);
    let mask = ~(0xFFFFu << shift);
    return (packed & mask) | (bf16 << shift);
}

struct SdpaParams {
    bs: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    q_seq: u32,
    k_seq: u32,
    head_dim: u32,
    v_dim: u32,
    gqa_factor: u32,
    scale_bits: u32,
    softcapping_bits: u32,
    has_mask: u32,
    do_causal: u32,
    ql_off: u32,
    _pad: array<u32, 59>,
}

@group(0) @binding(0)
var<storage, read_write> out_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> q_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> k_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> v_buf: array<u32>;

@group(0) @binding(4)
var<storage, read> sdpa_params: SdpaParams;

fn bitcast_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

fn q_index(bs_i: u32, qh: u32, qs: u32, d: u32) -> u32 {
    let hd = sdpa_params.head_dim;
    let qsh = sdpa_params.q_seq;
    let nqh = sdpa_params.n_q_heads;
    return ((bs_i * nqh + qh) * qsh + qs) * hd + d;
}

fn kv_index(bs_i: u32, kvh: u32, ks: u32, d: u32, stride_head: u32) -> u32 {
    return (bs_i * sdpa_params.n_kv_heads + kvh) * stride_head + ks * sdpa_params.head_dim + d;
}

fn v_index(bs_i: u32, kvh: u32, ks: u32, d: u32, stride_head: u32) -> u32 {
    return (bs_i * sdpa_params.n_kv_heads + kvh) * stride_head + ks * sdpa_params.v_dim + d;
}

fn out_index(bs_i: u32, qh: u32, qs: u32, d: u32) -> u32 {
    let vd = sdpa_params.v_dim;
    let qsh = sdpa_params.q_seq;
    let nqh = sdpa_params.n_q_heads;
    return ((bs_i * nqh + qh) * qsh + qs) * vd + d;
}

@compute @workgroup_size(1)
fn sdpa_vector_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let flat = wg_id.x;
    let q_seq = sdpa_params.q_seq;
    let n_q_heads = sdpa_params.n_q_heads;
    let per_bs = n_q_heads * q_seq;
    let bs_i = flat / per_bs;
    let rem = flat % per_bs;
    let qh = rem / q_seq;
    let qs = rem % q_seq;

    if (bs_i >= sdpa_params.bs) {
        return;
    }

    let head_dim = sdpa_params.head_dim;
    let v_dim = sdpa_params.v_dim;
    let k_seq = sdpa_params.k_seq;
    let gqa = sdpa_params.gqa_factor;
    let kvh = qh / gqa;
    let scale = bitcast_f32(sdpa_params.scale_bits);
    let softcapping = bitcast_f32(sdpa_params.softcapping_bits);

    let k_stride_head = k_seq * head_dim;
    let v_stride_head = k_seq * v_dim;

    var max_score = -1e38;
    var sum_exp = 0.0;
    var acc: array<f32, 128>;

    for (var d = 0u; d < v_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var ki = 0u; ki < k_seq; ki = ki + 1u) {
        var score = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            let qi = q_index(bs_i, qh, qs, d);
            let ki_idx = kv_index(bs_i, kvh, ki, d, k_stride_head);
            let qv = read_packed_bf16(q_buf[qi / 2u], qi);
            let kv = read_packed_bf16(k_buf[ki_idx / 2u], ki_idx);
            score += qv * kv;
        }
        score = score * scale;
        if (softcapping != 1.0) {
            score = tanh(score) * softcapping;
        }

        let new_max = max(max_score, score);
        let factor = exp(max_score - new_max);
        let exp_score = exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        for (var d = 0u; d < v_dim; d = d + 1u) {
            let vi = v_index(bs_i, kvh, ki, d, v_stride_head);
            let vv = read_packed_bf16(v_buf[vi / 2u], vi);
            acc[d] = acc[d] * factor + exp_score * vv;
        }
    }

    let inv_sum = 1.0 / sum_exp;
    for (var d = 0u; d < v_dim; d = d + 1u) {
        let oi = out_index(bs_i, qh, qs, d);
        out_buf[oi / 2u] = pack_bf16_value(out_buf[oi / 2u], oi, acc[d] * inv_sum);
    }
}
