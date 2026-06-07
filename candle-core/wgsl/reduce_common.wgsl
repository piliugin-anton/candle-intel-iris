// Reduction kernel shared definitions.
//
// Matches the standard bind group layout (bindings 0–3) but uses reduction-specific
// uniform fields. `in1` is unused; binding 2 still must be bound (duplicate `input0`).

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
var<storage, read_write> reduce_out_f32: array<f32>;

@group(0) @binding(1)
var<storage, read> reduce_in: array<f32>;

@group(0) @binding(2)
var<storage, read> reduce_in_pad: array<f32>;

@group(0) @binding(3)
var<storage, read> reduce_params: ReduceParams;

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

fn is_contiguous(tensor_layout: TensorLayout) -> bool {
    var acc = 1u;
    let num_dims = tensor_layout.num_dims;
    for (var d = 0u; d < num_dims; d = d + 1u) {
        let dim_idx = num_dims - 1u - d;
        let dim = tensor_layout.dims[dim_idx];
        if (dim > 1u && acc != tensor_layout.strides[dim_idx]) {
            return false;
        }
        acc *= dim;
    }
    return true;
}

fn reduce_dim_is_inner_contiguous() -> bool {
    let src = reduce_params.src_layout;
    let dim = reduce_params._pad0;
    if (!is_contiguous(src)) {
        return false;
    }
    if (dim + 1u != src.num_dims) {
        return false;
    }
    return src.strides[dim] == 1u;
}

fn reduce_src_base(dst_id: u32) -> u32 {
    return reduce_src_index(dst_id, 0u);
}

fn reduce_sum_inner_contiguous(dst_id: u32, chunk: u32, tid: u32) -> f32 {
    let base = reduce_src_base(dst_id);
    var acc = 0.0;
    var chunk_off = tid * 4u;
    let chunk_vec = (chunk / 4u) * 4u;
    while (chunk_off < chunk_vec) {
        let off = base + chunk_off;
        let v = vec4<f32>(
            reduce_in[off],
            reduce_in[off + 1u],
            reduce_in[off + 2u],
            reduce_in[off + 3u],
        );
        acc += v.x + v.y + v.z + v.w;
        chunk_off += REDUCE_WG_SIZE * 4u;
    }
    while (chunk_off < chunk) {
        acc += reduce_in[base + chunk_off];
        chunk_off += REDUCE_WG_SIZE;
    }
    return acc;
}

fn reduce_load_src(dst_id: u32, chunk_offset: u32) -> f32 {
    if (reduce_dim_is_inner_contiguous()) {
        return reduce_in[reduce_src_base(dst_id) + chunk_offset];
    }
    return reduce_in[reduce_src_index(dst_id, chunk_offset)];
}

// Flat workgroup index when dispatch is split across grid axes (max 65535 per dim).
fn reduce_dst_id(wg_id: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * num_wg.x + wg_id.z * num_wg.x * num_wg.y;
}

// Tree reduction in workgroup shared memory.
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
