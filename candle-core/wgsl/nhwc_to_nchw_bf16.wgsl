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
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: NhwcToNchwParams;

fn permute_value_at(dst_i: u32, p: NhwcToNchwParams) -> f32 {
    let w = p.w;
    let hw = p.h * w;
    let chw = p.c * hw;
    let b_idx = dst_i / chw;
    let rem = dst_i % chw;
    let c_idx = rem / hw;
    let rem2 = rem % hw;
    let h_idx = rem2 / w;
    let w_idx = rem2 % w;
    let src_i = p.src_offset + b_idx * hw * p.c + h_idx * w * p.c + w_idx * p.c + c_idx;
    return read_packed_bf16(input_buf[src_i / 2u], src_i);
}

@compute @workgroup_size(WG_SIZE)
fn nhwc_to_nchw_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = WG_SIZE * num_wg.x;
    let p = params;
    let dst_base = p.dst_offset;

    // Pair-packed writes at dst_base (always even for contiguous conv outputs).
    let num_words = (p.elem_count + 1u) / 2u;
    for (var pair = gid.x; pair < num_words; pair = pair + stride_wg) {
        let dst_i0 = pair * 2u;
        let phys0 = dst_base + dst_i0;
        let bf0 = f32_to_bf16_bits(permute_value_at(dst_i0, p));
        var packed = bf0;
        if (dst_i0 + 1u < p.elem_count) {
            let bf1 = f32_to_bf16_bits(permute_value_at(dst_i0 + 1u, p));
            packed = bf0 | (bf1 << 16u);
        }
        output_buf[phys0 / 2u] = packed;
    }
}
