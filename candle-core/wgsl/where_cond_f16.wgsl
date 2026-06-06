// where_cond with u8 predicate and f16 branches.
//
// Entry point: where_u8_f16

enable f16;

struct WhereParams {
    elem_count: u32,
    _pad: array<u32, 71>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> cond_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> on_true_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> on_false_buf: array<f16>;

@group(0) @binding(4)
var<storage, read> where_params: WhereParams;

fn load_cond_u8(idx: u32) -> u32 {
    let word = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    return (cond_buf[word] >> shift) & 0xFFu;
}

@compute @workgroup_size(32)
fn where_u8_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = 32u * num_wg.x;
    let count = where_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        if (load_cond_u8(i) != 0u) {
            output_buf[i] = on_true_buf[i];
        } else {
            output_buf[i] = on_false_buf[i];
        }
    }
}
