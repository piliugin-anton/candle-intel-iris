enable f16;
// Fused softmax along the last dimension (contiguous f16 input).
//
// Entry point: softmax_last_dim_f16

const WG_SIZE: u32 = 32u;

struct SoftmaxParams {
    n_rows: u32,
    last_dim: u32,
    _pad: array<u32, 62>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<f16>;

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
fn softmax_last_dim_f16(
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
        local_max = max(local_max, f32(input_buf[offset + i]));
    }
    wg_max[local_id] = local_max;
    workgroupBarrier();
    wg_reduce_max(local_id);
    workgroupBarrier();
    let row_max = wg_max[0];

    var local_sum = 0.0;
    for (var i = local_id; i < last_dim; i = i + WG_SIZE) {
        local_sum += exp(f32(input_buf[offset + i]) - row_max);
    }
    wg_sum[local_id] = local_sum;
    workgroupBarrier();
    wg_reduce_sum(local_id);
    workgroupBarrier();
    let row_sum = wg_sum[0];

    for (var i = local_id; i < last_dim; i = i + WG_SIZE) {
        let v = exp(f32(input_buf[offset + i]) - row_max) / row_sum;
        output_buf[offset + i] = f16(v);
    }
}
