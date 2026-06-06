// Fused scaled dot-product attention (decode / vector path, f32).
//
// Entry point: sdpa_vector_f32
//
// One workgroup per (batch, q_head, q_position). Threads cooperatively
// compute the Q·K dot product and update the value accumulator.

const SDPA_WG: u32 = 32u;
const MAX_V_DIM: u32 = 256u;

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
var<storage, read_write> out_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> q_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> k_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> v_buf: array<f32>;

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
fn sdpa_vector_f32(
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
            let qv = q_buf[q_index(bs_i, qh, qs, d)];
            let kv = k_buf[kv_index(bs_i, kvh, ki, d, k_stride_head)];
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
            let vv = v_buf[v_index(bs_i, kvh, ki, d, v_stride_head)];
            wg_acc[d] = wg_acc[d] * factor + exp_score * vv;
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        wg_inv_sum = 1.0 / wg_sum_exp;
    }
    workgroupBarrier();

    let inv_sum = wg_inv_sum;
    for (var d = lid; d < v_dim; d = d + SDPA_WG) {
        out_buf[out_index(bs_i, qh, qs, d)] = wg_acc[d] * inv_sum;
    }
}
