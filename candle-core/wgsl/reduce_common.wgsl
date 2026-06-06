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

fn reduce_get_strided_index(idx: u32, tensor_layout: TensorLayout) -> u32 {
    var remaining = idx;
    var strided_i = 0u;
    let num_dims = tensor_layout.num_dims;
    for (var d = 0u; d < num_dims; d = d + 1u) {
        let dim_idx = num_dims - 1u - d;
        let dim = tensor_layout.dims[dim_idx];
        let stride = tensor_layout.strides[dim_idx];
        strided_i += (remaining % dim) * stride;
        remaining /= dim;
    }
    return tensor_layout.offset + strided_i;
}

fn reduce_load_src(linear_idx: u32) -> f32 {
    // Buffer bindings include the layout byte offset; flat index matches contiguous tensors.
    return reduce_in[linear_idx];
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
