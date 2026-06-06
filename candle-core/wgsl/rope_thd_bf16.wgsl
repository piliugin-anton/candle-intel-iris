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

struct RopeThdParams {
    b: u32,
    t: u32,
    h: u32,
    d: u32,
    stride_b: u32,
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
var<storage, read> rope_thd_params: RopeThdParams;

@compute @workgroup_size(32)
fn rope_thd_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = rope_thd_params.b;
    let t = rope_thd_params.t;
    let h = rope_thd_params.h;
    let d = rope_thd_params.d;
    let stride_b = rope_thd_params.stride_b;
    let idx = gid.x;
    if (2u * idx >= b * t * h * d) {
        return;
    }

    let i_bth = idx / (d / 2u);
    let i_d = idx - (d / 2u) * i_bth;
    let i_t = (i_bth / h) % t;
    let i1 = i_bth * d + i_d;
    let i2 = i1 + d / 2u;
    var i_cs = i_t * (d / 2u) + i_d;
    if (stride_b > 0u) {
        let b_idx = (2u * idx) / stride_b;
        i_cs += b_idx * ((t * d) / 2u);
    }
    let c = read_packed_bf16(cos_buf[i_cs / 2u], i_cs);
    let s = read_packed_bf16(sin_buf[i_cs / 2u], i_cs);
    let v1 = read_packed_bf16(src_buf[i1 / 2u], i1);
    let v2 = read_packed_bf16(src_buf[i2 / 2u], i2);
    output_buf[i1 / 2u] = pack_bf16_value(output_buf[i1 / 2u], i1, v1 * c - v2 * s);
    output_buf[i2 / 2u] = pack_bf16_value(output_buf[i2 / 2u], i2, v1 * s + v2 * c);
}
