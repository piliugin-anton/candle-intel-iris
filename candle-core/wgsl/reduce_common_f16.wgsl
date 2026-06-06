enable f16;
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
var<storage, read_write> reduce_out: array<f16>;

@group(0) @binding(1)
var<storage, read> reduce_in: array<f16>;

@group(0) @binding(2)
var<storage, read> reduce_in_pad: array<f16>;

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

fn reduce_load_src(dst_id: u32, chunk_offset: u32) -> f16 {
    return reduce_in[reduce_src_index(dst_id, chunk_offset)];
}

// Tree reduction in workgroup shared memory.
var<workgroup> wg_sum: array<f16, REDUCE_WG_SIZE>;
var<workgroup> wg_max_val: array<f16, REDUCE_WG_SIZE>;
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
