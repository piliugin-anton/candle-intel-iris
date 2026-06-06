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

struct RopeIParams {
    bh: u32,
    td: u32,
    stride_b: u32,
    _pad: array<u32, 68>,
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
var<storage, read> rope_i_params: RopeIParams;

@compute @workgroup_size(32)
fn rope_i_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bh = rope_i_params.bh;
    let td = rope_i_params.td;
    let stride_b = rope_i_params.stride_b;
    let idx = gid.x;
    if (2u * idx >= bh * td) {
        return;
    }

    var rope_idx = idx % (td / 2u);
    if (stride_b > 0u) {
        let b_idx = (2u * idx) / stride_b;
        rope_idx += b_idx * (td / 2u);
    }
    let c = read_packed_bf16(cos_buf[rope_idx / 2u], rope_idx);
    let s = read_packed_bf16(sin_buf[rope_idx / 2u], rope_idx);
    let i0 = 2u * idx;
    let i1 = i0 + 1u;
    let v0 = read_packed_bf16(src_buf[i0 / 2u], i0);
    let v1 = read_packed_bf16(src_buf[i1 / 2u], i1);
    output_buf[i0 / 2u] = pack_bf16_value(output_buf[i0 / 2u], i0, v0 * c - v1 * s);
    output_buf[i1 / 2u] = pack_bf16_value(output_buf[i1 / 2u], i1, v0 * s + v1 * c);
}
