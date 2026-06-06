// Fused softmax along the last dimension (contiguous bf16 input).
//
// Entry point: softmax_last_dim_bf16

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

const WG_SIZE: u32 = 32u;

struct SoftmaxParams {
    n_rows: u32,
    last_dim: u32,
    _pad: array<u32, 62>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> softmax_params: SoftmaxParams;

var<workgroup> wg_max: array<f32, WG_SIZE>;
var<workgroup> wg_sum: array<f32, WG_SIZE>;

fn wg_reduce_max(local_id: u32) {
    var stride = WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            wg_max[local_id] = max(wg_max[local_id], wg_max[local_id + stride]);
        }
        stride = stride / 2u;
    }
}

fn wg_reduce_sum(local_id: u32) {
    var stride = WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            wg_sum[local_id] = wg_sum[local_id] + wg_sum[local_id + stride];
        }
        stride = stride / 2u;
    }
}

@compute @workgroup_size(WG_SIZE)
fn softmax_last_dim_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x;
    if (row >= softmax_params.n_rows) {
        return;
    }

    let last_dim = softmax_params.last_dim;
    let offset = row * last_dim;
    let local_id = lid.x;

    var local_max = -1e38;
    for (var i = local_id; i < last_dim; i = i + WG_SIZE) {
        let idx = offset + i;
        local_max = max(local_max, read_packed_bf16(input_buf[idx / 2u], idx));
    }
    wg_max[local_id] = local_max;
    workgroupBarrier();
    wg_reduce_max(local_id);
    workgroupBarrier();
    let row_max = wg_max[0];

    var local_sum = 0.0;
    for (var i = local_id; i < last_dim; i = i + WG_SIZE) {
        let idx = offset + i;
        local_sum += exp(read_packed_bf16(input_buf[idx / 2u], idx) - row_max);
    }
    wg_sum[local_id] = local_sum;
    workgroupBarrier();
    wg_reduce_sum(local_id);
    workgroupBarrier();
    let row_sum = wg_sum[0];

    // Packed bf16 writes are not race-safe across threads in one workgroup.
    if (local_id == 0u) {
        for (var i = 0u; i < last_dim; i = i + 1u) {
            let idx = offset + i;
            let v = exp(read_packed_bf16(input_buf[idx / 2u], idx) - row_max) / row_sum;
            output_buf[idx / 2u] = pack_bf16_value(output_buf[idx / 2u], idx, v);
        }
    }
}
