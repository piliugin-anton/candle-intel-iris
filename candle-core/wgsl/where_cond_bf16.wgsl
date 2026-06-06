// where_cond with u8 predicate and packed bf16 branches.
//
// Entry point: where_u8_bf16

struct WhereParams {
    elem_count: u32,
    _pad: array<u32, 71>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> cond_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> on_true_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> on_false_buf: array<u32>;

@group(0) @binding(4)
var<storage, read> where_params: WhereParams;

fn load_cond_u8(idx: u32) -> u32 {
    let word = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    return (cond_buf[word] >> shift) & 0xFFu;
}

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn load_bf16_true(idx: u32) -> f32 {
    let word = idx / 2u;
    let byte_off = (idx % 2u) * 2u;
    let packed = on_true_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn load_bf16_false(idx: u32) -> f32 {
    let word = idx / 2u;
    let byte_off = (idx % 2u) * 2u;
    let packed = on_false_buf[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn store_bf16_elem(idx: u32, value: f32) {
    let word = idx / 2u;
    let byte_off = (idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = f32_to_bf16_bits(value);
    let mask = ~(0xFFFFu << shift);
    let contribution = bf16 << shift;
    var old = atomicLoad(&output_buf[word]);
    loop {
        let new_val = (old & mask) | contribution;
        let exch = atomicCompareExchangeWeak(&output_buf[word], old, new_val);
        if (exch.exchanged) {
            break;
        }
        old = exch.old_value;
    }
}

@compute @workgroup_size(32)
fn where_u8_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = 32u * num_wg.x;
    let count = where_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        if (load_cond_u8(i) != 0u) {
            store_bf16_elem(i, load_bf16_true(i));
        } else {
            store_bf16_elem(i, load_bf16_false(i));
        }
    }
}
