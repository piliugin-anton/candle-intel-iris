enable f16;
// Shared Candle wgpu kernel utilities.
//
// Matches Rust `TensorLayoutUniform` and `KernelUniforms` in
// `candle-core/src/wgpu_device/bind_group.rs`.

const MAX_DIMS: u32 = 8u;

// Intel integrated GPUs (e.g. Iris) perform well with 8–32-wide workgroups.
const WG_SIZE: u32 = 32u;

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
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> kernel_params: KernelParams;

// Convert a flat logical index into a physical buffer offset for a strided tensor.
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

// Row-major contiguous check (same semantics as CUDA `is_contiguous`).
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

fn buffer_index(idx: u32, tensor_layout: TensorLayout) -> u32 {
    if (is_contiguous(tensor_layout)) {
        return tensor_layout.offset + idx;
    }
    return get_strided_index(idx, tensor_layout);
}

const BROADCAST_BIAS_ADD: u32 = 1u;

fn nchw_coords(idx: u32, tensor_layout: TensorLayout) -> vec4<u32> {
    let w = tensor_layout.dims[3];
    let hw = tensor_layout.dims[2] * w;
    let chw = tensor_layout.dims[1] * hw;
    let b_idx = idx / chw;
    let rem = idx % chw;
    let c_idx = rem / hw;
    let rem2 = rem % hw;
    let h_idx = rem2 / w;
    let w_idx = rem2 % w;
    return vec4<u32>(b_idx, c_idx, h_idx, w_idx);
}

fn buffer_index_bias_in0(idx: u32) -> u32 {
    let in0 = kernel_params.in0_layout;
    let coords = nchw_coords(idx, kernel_params.out_layout);
    return in0.offset
        + coords.x * in0.strides[0]
        + coords.z * in0.strides[2]
        + coords.w * in0.strides[3];
}

fn buffer_index_bias_in1(idx: u32) -> u32 {
    let in1 = kernel_params.in1_layout;
    let coords = nchw_coords(idx, kernel_params.out_layout);
    return in1.offset + coords.x * in1.strides[0] + coords.y * in1.strides[1];
}

fn fused_bias_add_f16(idx: u32) -> f16 {
    let coords = nchw_coords(idx, kernel_params.out_layout);
    let in0 = kernel_params.in0_layout;
    let in1 = kernel_params.in1_layout;
    let i0 = in0.offset + coords.x * in0.strides[0] + coords.z * in0.strides[2] + coords.w;
    let i1 = in1.offset + coords.x * in1.strides[0] + coords.y * in1.strides[1];
    return input0_buf[i0] + input1_buf[i1];
}

fn load_in0(idx: u32) -> f16 {
    if (kernel_params._pad2 == BROADCAST_BIAS_ADD) {
        return input0_buf[buffer_index_bias_in0(idx)];
    }
    return input0_buf[buffer_index(idx, kernel_params.in0_layout)];
}

fn load_in1(idx: u32) -> f16 {
    if (kernel_params._pad2 == BROADCAST_BIAS_ADD) {
        return input1_buf[buffer_index_bias_in1(idx)];
    }
    return input1_buf[buffer_index(idx, kernel_params.in1_layout)];
}

fn store_out(idx: u32, value: f16) {
    let out_layout = kernel_params.out_layout;
    if (is_contiguous(out_layout)) {
        output_buf[out_layout.offset + idx] = value;
    } else {
        output_buf[get_strided_index(idx, out_layout)] = value;
    }
}

// Total thread count along the X grid axis (for grid-stride element loops).
fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}

