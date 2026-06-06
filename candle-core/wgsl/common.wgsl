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
    _pad: vec3<u32>,
    out_layout: TensorLayout,
    in0_layout: TensorLayout,
    in1_layout: TensorLayout,
    _tail_pad: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> input1_buf: array<f32>;

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

// Bindings include the layout byte offset; output indices are contiguous row-major.
fn load_in0(idx: u32) -> f32 {
    return input0_buf[idx];
}

fn load_in1(idx: u32) -> f32 {
    return input1_buf[idx];
}

fn store_out(idx: u32, value: f32) {
    output_buf[idx] = value;
}

// Total thread count along the X grid axis (for grid-stride element loops).
fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}
