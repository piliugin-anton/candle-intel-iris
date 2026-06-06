enable f16;
// Fused scaled dot-product attention (prefill / full path, f16).
//
// Entry point: sdpa_full_f16

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
var<storage, read_write> out_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> q_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> k_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> v_buf: array<f16>;

@group(0) @binding(4)
var<storage, read> mask_buf: array<f16>;

@group(0) @binding(5)
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

fn mask_index(bs_i: u32, qh: u32, qs: u32, ki: u32) -> u32 {
    let ksh = sdpa_params.k_seq;
    let qsh = sdpa_params.q_seq;
    let nqh = sdpa_params.n_q_heads;
    return ((bs_i * nqh + qh) * qsh + qs) * ksh + ki;
}

fn is_masked(bs_i: u32, qh: u32, qs: u32, ki: u32) -> bool {
    if (sdpa_params.do_causal != 0u && ki > qs + sdpa_params.ql_off) {
        return true;
    }
    if (sdpa_params.has_mask != 0u) {
        let m = f32(mask_buf[mask_index(bs_i, qh, qs, ki)]);
        if (m < -1e30) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(1)
fn sdpa_full_f16(@builtin(workgroup_id) wg_id: vec3<u32>) {
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
    var acc: array<f32, 256>;

    for (var d = 0u; d < v_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var ki = 0u; ki < k_seq; ki = ki + 1u) {
        if (is_masked(bs_i, qh, qs, ki)) {
            continue;
        }

        var score = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            let qv = f32(q_buf[q_index(bs_i, qh, qs, d)]);
            let kv = f32(k_buf[kv_index(bs_i, kvh, ki, d, k_stride_head)]);
            score += qv * kv;
        }
        score = score * scale;
        if (softcapping != 1.0) {
            score = tanh(score) * softcapping;
        }
        if (sdpa_params.has_mask != 0u) {
            score += f32(mask_buf[mask_index(bs_i, qh, qs, ki)]);
        }

        let new_max = max(max_score, score);
        let factor = exp(max_score - new_max);
        let exp_score = exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        for (var d = 0u; d < v_dim; d = d + 1u) {
            let vv = f32(v_buf[v_index(bs_i, kvh, ki, d, v_stride_head)]);
            acc[d] = acc[d] * factor + exp_score * vv;
        }
    }

    let inv_sum = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);
    for (var d = 0u; d < v_dim; d = d + 1u) {
        out_buf[out_index(bs_i, qh, qs, d)] = f16(acc[d] * inv_sum);
    }
}
