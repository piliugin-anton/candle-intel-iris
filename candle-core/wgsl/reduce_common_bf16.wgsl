// Reduction kernel shared definitions for packed bf16 (f32 accumulation).

const MAX_DIMS: u32 = 8u;
const REDUCE_WG_SIZE: u32 = 32u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct ReduceParams {
    src_elem_count: u32,
    dst_elem_count: u32,
    reduce_chunk_size: u32,
    _pad0: u32,
    out_layout: TensorLayout,
    src_layout: TensorLayout,
    _unused_layout: TensorLayout,
    _tail_pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> reduce_out: array<u32>;

@group(0) @binding(1)
var<storage, read> reduce_in: array<u32>;

@group(0) @binding(2)
var<storage, read> reduce_in_pad: array<u32>;

@group(0) @binding(3)
var<storage, read> reduce_params: ReduceParams;

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16_bits(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn reduce_src_index(dst_id: u32, chunk_offset: u32) -> u32 {
    let out_layout = reduce_params.out_layout;
    let src_layout = reduce_params.src_layout;
    let chunk = reduce_params.reduce_chunk_size;
    let num_dims = out_layout.num_dims;
    var remaining = dst_id;
    var src_idx = src_layout.offset;
    for (var d = 0u; d < num_dims; d = d + 1u) {
        let dim_idx = num_dims - 1u - d;
        let out_dim = out_layout.dims[dim_idx];
        let coord = remaining % out_dim;
        remaining = remaining / out_dim;
        let src_dim = src_layout.dims[dim_idx];
        var src_coord = coord;
        if (out_dim == 1u && src_dim == chunk) {
            src_coord = chunk_offset;
        }
        src_idx += src_coord * src_layout.strides[dim_idx];
    }
    return src_idx;
}

fn reduce_load_src(dst_id: u32, chunk_offset: u32) -> f32 {
    let elem = reduce_src_index(dst_id, chunk_offset);
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
    let packed = reduce_in[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return bf16_bits_to_f32(bf16);
}

fn reduce_store_out(dst_id: u32, value: f32) {
    let elem = dst_id + reduce_params.out_layout.offset;
    let word = elem / 2u;
    let byte_off = (elem % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = f32_to_bf16_bits(value);
    let mask = ~(0xFFFFu << shift);
    var packed = reduce_out[word];
    packed = (packed & mask) | (bf16 << shift);
    reduce_out[word] = packed;
}

fn reduce_dst_id(wg_id: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * num_wg.x + wg_id.z * num_wg.x * num_wg.y;
}

var<workgroup> wg_sum: array<f32, REDUCE_WG_SIZE>;
var<workgroup> wg_max_val: array<f32, REDUCE_WG_SIZE>;
var<workgroup> wg_max_idx: array<u32, REDUCE_WG_SIZE>;

fn workgroup_reduce_sum(local_id: u32) {
    var stride = REDUCE_WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            wg_sum[local_id] += wg_sum[local_id + stride];
        }
        stride /= 2u;
    }
}

fn workgroup_reduce_max(local_id: u32) {
    var stride = REDUCE_WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            let other = wg_max_val[local_id + stride];
            if (other > wg_max_val[local_id]) {
                wg_max_val[local_id] = other;
                wg_max_idx[local_id] = wg_max_idx[local_id + stride];
            }
        }
        stride /= 2u;
    }
}

fn workgroup_reduce_min(local_id: u32) {
    var stride = REDUCE_WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            wg_max_val[local_id] = min(wg_max_val[local_id], wg_max_val[local_id + stride]);
        }
        stride /= 2u;
    }
}

fn workgroup_reduce_argmin(local_id: u32) {
    var stride = REDUCE_WG_SIZE / 2u;
    while (stride > 0u) {
        workgroupBarrier();
        if (local_id < stride) {
            let other = wg_max_val[local_id + stride];
            if (other < wg_max_val[local_id]) {
                wg_max_val[local_id] = other;
                wg_max_idx[local_id] = wg_max_idx[local_id + stride];
            }
        }
        stride /= 2u;
    }
}
