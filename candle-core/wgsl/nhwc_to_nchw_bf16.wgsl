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

fn write_packed_bf16(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
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

// Permute a contiguous BHWC tensor into contiguous BCHW (NCHW) for packed bf16.
//
// Entry point: nhwc_to_nchw_bf16

const WG_SIZE: u32 = 32u;

struct NhwcToNchwParams {
    elem_count: u32,
    b: u32,
    h: u32,
    w: u32,
    c: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: array<u32, 65>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: NhwcToNchwParams;

@compute @workgroup_size(WG_SIZE)
fn nhwc_to_nchw_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = WG_SIZE * num_wg.x;
    let p = params;
    let w = p.w;
    let hw = p.h * w;
    let chw = p.c * hw;
    let src_base = p.src_offset;
    let dst_base = p.dst_offset;

    for (var dst_i = gid.x; dst_i < p.elem_count; dst_i = dst_i + stride_wg) {
        let b_idx = dst_i / chw;
        let rem = dst_i % chw;
        let c_idx = rem / hw;
        let rem2 = rem % hw;
        let h_idx = rem2 / w;
        let w_idx = rem2 % w;
        let src_i = src_base + b_idx * hw * p.c + h_idx * w * p.c + w_idx * p.c + c_idx;
        let value = read_packed_bf16(input_buf[src_i / 2u], src_i);
        write_packed_bf16(dst_base + dst_i, value);
    }
}
