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
const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct UpsampleNearest1dParams {
    dst_sz: u32,
    scale_bits: u32,
    dst_numel: u32,
    _align: u32,
    src_layout: TensorLayout,
    _pad: array<u32, 48>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: UpsampleNearest1dParams;

@compute @workgroup_size(32)
fn upsample_nearest1d_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = 32u * num_wg.x;
    let p = params;
    let src_layout = p.src_layout;
    let dst_sz = p.dst_sz;
    let src_sz = src_layout.dims[2];
    let scale = bitcast<f32>(p.scale_bits);
    let c = src_layout.dims[1];

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (c * dst_sz);
        let c_idx = (tid / dst_sz) % c;
        let dst_idx = tid % dst_sz;
        let src_idx = min(src_sz - 1u, u32(f32(dst_idx) * scale));
        let src_i = src_layout.offset
            + b_idx * src_layout.strides[0]
            + c_idx * src_layout.strides[1]
            + src_idx * src_layout.strides[2];
        output_buf[tid / 2u] = pack_bf16_value(output_buf[tid / 2u], tid, read_packed_bf16(input_buf[src_i / 2u], src_i));
    }
}
