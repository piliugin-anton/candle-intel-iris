// Indexing kernels: index_select, gather, scatter, scatter_add, index_add.
//
// Data tensors are f32. Index tensors are u32 or u8 (byte-packed in u32 words).
// Matches CUDA `indexing.cu` / Metal `indexing.metal` semantics.

const WG_SIZE: u32 = 32u;
const U32_MAX: u32 = 0xFFFFFFFFu;
const U8_MAX: u32 = 0xFFu;

struct IndexingParams {
    elem_count: u32,
    left_size: u32,
    src_dim_size: u32,
    dim_size: u32,
    right_size: u32,
    ids_dim_size: u32,
    _pad: array<u32, 66>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> ids_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> params: IndexingParams;

fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}

fn load_id_u32(flat_idx: u32) -> u32 {
    return ids_buf[flat_idx];
}

fn load_id_u8(flat_idx: u32) -> u32 {
    let word = flat_idx >> 2u;
    let shift = (flat_idx & 3u) * 8u;
    return (ids_buf[word] >> shift) & 0xFFu;
}

@compute @workgroup_size(WG_SIZE)
fn index_select_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let ids_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var dst_i = gid.x; dst_i < count; dst_i = dst_i + stride) {
        let left_i = dst_i / (ids_dim_size * right_size);
        let id_i = dst_i / right_size % ids_dim_size;
        let right_i = dst_i % right_size;
        let id = load_id_u32(id_i);
        if (id == U32_MAX) {
            output_buf[dst_i] = 0.0;
        } else {
            let src_i = left_i * (src_dim_size * right_size) + id * right_size + right_i;
            output_buf[dst_i] = input_buf[src_i];
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn index_select_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let ids_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var dst_i = gid.x; dst_i < count; dst_i = dst_i + stride) {
        let left_i = dst_i / (ids_dim_size * right_size);
        let id_i = dst_i / right_size % ids_dim_size;
        let right_i = dst_i % right_size;
        let id = load_id_u8(id_i);
        if (id == U8_MAX) {
            output_buf[dst_i] = 0.0;
        } else {
            let src_i = left_i * (src_dim_size * right_size) + id * right_size + right_i;
            output_buf[dst_i] = input_buf[src_i];
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn gather_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let ids_dim_size = params.dim_size;
    let ids_right_size = params.right_size;
    let src_right_size = params.ids_dim_size;

    for (var dst_i = gid.x; dst_i < count; dst_i = dst_i + stride) {
        let ids_right_i = dst_i % ids_right_size;
        let tmp = dst_i / ids_right_size;
        let left_i = tmp / ids_dim_size;
        let id = load_id_u32(dst_i);
        if (id == U32_MAX) {
            output_buf[dst_i] = 0.0;
        } else {
            let src_i = left_i * src_dim_size * src_right_size
                + id * src_right_size
                + ids_right_i;
            output_buf[dst_i] = input_buf[src_i];
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn gather_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let ids_dim_size = params.dim_size;
    let ids_right_size = params.right_size;
    let src_right_size = params.ids_dim_size;

    for (var dst_i = gid.x; dst_i < count; dst_i = dst_i + stride) {
        let ids_right_i = dst_i % ids_right_size;
        let tmp = dst_i / ids_right_size;
        let left_i = tmp / ids_dim_size;
        let id = load_id_u8(dst_i);
        if (id == U8_MAX) {
            output_buf[dst_i] = 0.0;
        } else {
            let src_i = left_i * src_dim_size * src_right_size
                + id * src_right_size
                + ids_right_i;
            output_buf[dst_i] = input_buf[src_i];
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn scatter_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let right_rank_i = tid % right_size;
        let left_rank_i = tid / right_size;
        for (var j = 0u; j < src_dim_size; j = j + 1u) {
            let src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
            let idx = load_id_u32(src_i);
            if (idx < U32_MAX) {
                let dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
                output_buf[dst_i] = input_buf[src_i];
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn scatter_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let right_rank_i = tid % right_size;
        let left_rank_i = tid / right_size;
        for (var j = 0u; j < src_dim_size; j = j + 1u) {
            let src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
            let idx = load_id_u8(src_i);
            if (idx < U8_MAX) {
                let dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
                output_buf[dst_i] = input_buf[src_i];
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn scatter_add_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let right_rank_i = tid % right_size;
        let left_rank_i = tid / right_size;
        for (var j = 0u; j < src_dim_size; j = j + 1u) {
            let src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
            let idx = load_id_u32(src_i);
            if (idx < U32_MAX) {
                let dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
                output_buf[dst_i] = output_buf[dst_i] + input_buf[src_i];
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn scatter_add_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let right_rank_i = tid % right_size;
        let left_rank_i = tid / right_size;
        for (var j = 0u; j < src_dim_size; j = j + 1u) {
            let src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
            let idx = load_id_u8(src_i);
            if (idx < U8_MAX) {
                let dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
                output_buf[dst_i] = output_buf[dst_i] + input_buf[src_i];
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn index_add_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let ids_dim_size = params.ids_dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let pre = tid / right_size;
        let post = tid % right_size;
        for (var j = 0u; j < ids_dim_size; j = j + 1u) {
            let idx = load_id_u32(j);
            if (idx < U32_MAX) {
                let src_i = (pre * src_dim_size + j) * right_size + post;
                let dst_i = (pre * dst_dim_size + idx) * right_size + post;
                output_buf[dst_i] = output_buf[dst_i] + input_buf[src_i];
            }
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn index_add_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = params.elem_count;
    let src_dim_size = params.src_dim_size;
    let dst_dim_size = params.dim_size;
    let ids_dim_size = params.ids_dim_size;
    let right_size = params.right_size;

    for (var tid = gid.x; tid < count; tid = tid + stride) {
        let pre = tid / right_size;
        let post = tid % right_size;
        for (var j = 0u; j < ids_dim_size; j = j + 1u) {
            let idx = load_id_u8(j);
            if (idx < U8_MAX) {
                let src_i = (pre * src_dim_size + j) * right_size + post;
                let dst_i = (pre * dst_dim_size + idx) * right_size + post;
                output_buf[dst_i] = output_buf[dst_i] + input_buf[src_i];
            }
        }
    }
}
