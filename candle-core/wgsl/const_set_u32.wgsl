// Fill a (possibly strided) u32 tensor with a scalar.
//
// `_pad0` holds the u32 value (`KernelUniforms::new_const_set`).
// Entry point: const_set_u32

const WG_SIZE: u32 = 32u;
const MAX_DIMS: u32 = 8u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct KernelParams {
    elem_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    out_layout: TensorLayout,
    in0_layout: TensorLayout,
    in1_layout: TensorLayout,
    _tail_pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> kernel_params: KernelParams;

fn get_strided_index(idx: u32, tensor_layout: TensorLayout) -> u32 {
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

fn store_out(idx: u32, value: u32) {
    let out_layout = kernel_params.out_layout;
    if (is_contiguous(out_layout)) {
        output_buf[out_layout.offset + idx] = value;
    } else {
        output_buf[get_strided_index(idx, out_layout)] = value;
    }
}

fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}

@compute @workgroup_size(WG_SIZE)
fn const_set_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = grid_stride_x(num_wg);
    let count = kernel_params.elem_count;
    let value = kernel_params._pad0;

    for (var i = gid.x; i < count; i = i + stride) {
        store_out(i, value);
    }
}
