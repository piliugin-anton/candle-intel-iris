// NeoX-style rotary positional embedding (bf16).
//
// Entry point: rope_bf16

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

struct RopeParams {
    b: u32,
    h: u32,
    t: u32,
    d: u32,
    unbatched_cs: u32,
    _pad: array<u32, 67>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> src_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> cos_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> sin_buf: array<u32>;

@group(0) @binding(4)
var<storage, read> rope_params: RopeParams;

@compute @workgroup_size(32)
fn rope_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = rope_params.b;
    let h = rope_params.h;
    let t = rope_params.t;
    let d = rope_params.d;
    let half_d = d / 2u;
    let bh = b * h;
    let flat_bh = gid.x / t;
    let i_t = gid.x % t;
    if (flat_bh >= bh) {
        return;
    }

    let bh_i = flat_bh;
    let row_base = bh_i * t * d + i_t * d;
    for (var i_d = 0u; i_d < half_d; i_d = i_d + 1u) {
        let i1 = row_base + i_d;
        let i2 = i1 + half_d;
        var i_cs = i_t * half_d + i_d;
        if (rope_params.unbatched_cs == 0u) {
            let b_i = bh_i / h;
            i_cs += b_i * t * half_d;
        }
        let c = read_packed_bf16(cos_buf[i_cs / 2u], i_cs);
        let s = read_packed_bf16(sin_buf[i_cs / 2u], i_cs);
        let v1 = read_packed_bf16(src_buf[i1 / 2u], i1);
        let v2 = read_packed_bf16(src_buf[i2 / 2u], i2);
        output_buf[i1 / 2u] = pack_bf16_value(output_buf[i1 / 2u], i1, v1 * c - v2 * s);
        output_buf[i2 / 2u] = pack_bf16_value(output_buf[i2 / 2u], i2, v1 * s + v2 * c);
    }
}
