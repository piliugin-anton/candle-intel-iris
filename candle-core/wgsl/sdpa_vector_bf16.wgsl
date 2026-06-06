// Fused scaled dot-product attention (decode / vector path, bf16).

const SDPA_WG: u32 = 32u;
const MAX_V_DIM: u32 = 256u;

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

fn pack_bf16_pair(lo: f32, hi: f32) -> u32 {
    let lo_bits = f32_to_bf16_bits(lo);
    let hi_bits = f32_to_bf16_bits(hi);
    return lo_bits | (hi_bits << 16u);
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

var<workgroup> partial_dot: array<f32, SDPA_WG>;
var<workgroup> wg_acc: array<f32, MAX_V_DIM>;
var<workgroup> wg_max_score: f32;
var<workgroup> wg_sum_exp: f32;
var<workgroup> wg_factor: f32;
var<workgroup> wg_exp_score: f32;
var<workgroup> wg_inv_sum: f32;

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

@compute @workgroup_size(SDPA_WG)
fn sdpa_vector_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lid = local_id.x;
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

    for (var d = lid; d < v_dim; d = d + SDPA_WG) {
        wg_acc[d] = 0.0;
    }
    if (lid == 0u) {
        wg_max_score = -1e38;
        wg_sum_exp = 0.0;
    }
    workgroupBarrier();

    for (var ki = 0u; ki < k_seq; ki = ki + 1u) {
        var local_dot = 0.0;
        for (var d = lid; d < head_dim; d = d + SDPA_WG) {
            let qi = q_index(bs_i, qh, qs, d);
            let ki_idx = kv_index(bs_i, kvh, ki, d, k_stride_head);
            let qv = read_packed_bf16(q_buf[qi / 2u], qi);
            let kv = read_packed_bf16(k_buf[ki_idx / 2u], ki_idx);
            local_dot += qv * kv;
        }
        partial_dot[lid] = local_dot;
        workgroupBarrier();

        if (lid == 0u) {
            var score = 0.0;
            for (var i = 0u; i < SDPA_WG; i = i + 1u) {
                score += partial_dot[i];
            }
            score = score * scale;
            if (softcapping != 1.0) {
                score = tanh(score) * softcapping;
            }

            let new_max = max(wg_max_score, score);
            wg_factor = exp(wg_max_score - new_max);
            wg_exp_score = exp(score - new_max);
            wg_max_score = new_max;
            wg_sum_exp = wg_sum_exp * wg_factor + wg_exp_score;
        }
        workgroupBarrier();

        let factor = wg_factor;
        let exp_score = wg_exp_score;
        for (var d = lid; d < v_dim; d = d + SDPA_WG) {
            let vi = v_index(bs_i, kvh, ki, d, v_stride_head);
            let vv = read_packed_bf16(v_buf[vi / 2u], vi);
            wg_acc[d] = wg_acc[d] * factor + exp_score * vv;
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        wg_inv_sum = 1.0 / wg_sum_exp;
    }
    workgroupBarrier();

    let inv_sum = wg_inv_sum;
    let word_count = (v_dim + 1u) / 2u;
    for (var w = lid; w < word_count; w = w + SDPA_WG) {
        let d0 = w * 2u;
        let oi = out_index(bs_i, qh, qs, d0);
        let lo = wg_acc[d0] * inv_sum;
        var hi = 0.0;
        if (d0 + 1u < v_dim) {
            hi = wg_acc[d0 + 1u] * inv_sum;
        }
        out_buf[oi / 2u] = pack_bf16_pair(lo, hi);
    }
}
