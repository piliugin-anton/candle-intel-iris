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
struct Copy2dParams {
    d1: u32,
    d2: u32,
    src_stride: u32,
    dst_stride: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: array<u32, 66>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> copy2d_params: Copy2dParams;

@compute @workgroup_size(32)
fn copy2d_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = 32u * num_wg.x;
    let total = copy2d_params.d1 * copy2d_params.d2;
    let src_stride = copy2d_params.src_stride;
    let dst_stride = copy2d_params.dst_stride;
    let src_base = copy2d_params.src_offset;
    let dst_base = copy2d_params.dst_offset;
    let d2 = copy2d_params.d2;

    for (var flat = gid.x; flat < total; flat = flat + stride) {
        let row = flat / d2;
        let col = flat % d2;
        let src_idx = src_base + row * src_stride + col;
        let dst_idx = dst_base + row * dst_stride + col;
        output_buf[dst_idx / 2u] = pack_bf16_value(
            output_buf[dst_idx / 2u],
            dst_idx,
            read_packed_bf16(input0_buf[src_idx / 2u], src_idx),
        );
    }
}
