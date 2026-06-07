// Transposed 2D convolution (f32). Entry point: conv_transpose2d_f32

const MAX_DIMS: u32 = 8u;

const WG_SIZE: u32 = 32u;

struct TensorLayout {
    dims: array<u32, MAX_DIMS>,
    strides: array<u32, MAX_DIMS>,
    offset: u32,
    num_dims: u32,
    _pad: vec2<u32>,
}

struct ConvTranspose2dParams {
    w_out: u32,
    h_out: u32,
    stride: u32,
    padding: u32,
    output_padding: u32,
    dilation: u32,
    dst_numel: u32,
    _align: u32,
    src_layout: TensorLayout,
    kernel_layout: TensorLayout,
    _pad: array<u32, 24>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> src_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> kernel_buf: array<f32>;

@group(0) @binding(4)
var<storage, read> params: ConvTranspose2dParams;

@compute @workgroup_size(WG_SIZE)
fn conv_transpose2d_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride_wg = WG_SIZE * num_wg.x;
    let p = params;
    let in_layout = p.src_layout;
    let k_layout = p.kernel_layout;
    let h_k = k_layout.dims[2];
    let w_k = k_layout.dims[3];
    let c_out = k_layout.dims[1];
    let c_in = in_layout.dims[1];
    let h_in = in_layout.dims[2];
    let w_in = in_layout.dims[3];
    let w_out = p.w_out;
    let h_out = p.h_out;
    let stride = p.stride;
    let padding = p.padding;
    let dilation = p.dilation;

    for (var tid = gid.x; tid < p.dst_numel; tid = tid + stride_wg) {
        let b_idx = tid / (w_out * h_out * c_out);
        let dst_c_idx = (tid / (w_out * h_out)) % c_out;
        let out_y = (tid / w_out) % h_out;
        let out_x = tid % w_out;

        let src_idx0 = in_layout.offset + b_idx * in_layout.strides[0];
        var acc = 0.0;
        for (var k_x = 0; k_x < i32(w_k); k_x = k_x + 1) {
            let inp_x_stride = i32(out_x + padding) - k_x * i32(dilation);
            if (inp_x_stride < 0 || inp_x_stride % i32(stride) != 0) {
                continue;
            }
            let inp_x = inp_x_stride / i32(stride);
            if (inp_x >= i32(w_in)) {
                continue;
            }
            for (var k_y = 0; k_y < i32(h_k); k_y = k_y + 1) {
                let inp_y_stride = i32(out_y + padding) - k_y * i32(dilation);
                if (inp_y_stride < 0 || inp_y_stride % i32(stride) != 0) {
                    continue;
                }
                let inp_y = inp_y_stride / i32(stride);
                if (inp_y >= i32(h_in)) {
                    continue;
                }
                for (var src_c_idx = 0u; src_c_idx < c_in; src_c_idx = src_c_idx + 1u) {
                    let src_idx = src_idx0
                        + src_c_idx * in_layout.strides[1]
                        + u32(inp_y) * in_layout.strides[2]
                        + u32(inp_x) * in_layout.strides[3];
                    let k_idx = k_layout.offset
                        + src_c_idx * k_layout.strides[0]
                        + dst_c_idx * k_layout.strides[1]
                        + u32(k_y) * k_layout.strides[2]
                        + u32(k_x) * k_layout.strides[3];
                    acc += src_buf[src_idx] * kernel_buf[k_idx];
                }
            }
        }
        output_buf[tid] = acc;
    }
}
