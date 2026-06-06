enable f16;

struct RopeIParams {
    bh: u32,
    td: u32,
    stride_b: u32,
    _pad: array<u32, 68>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f16>;

@group(0) @binding(1)
var<storage, read> src_buf: array<f16>;

@group(0) @binding(2)
var<storage, read> cos_buf: array<f16>;

@group(0) @binding(3)
var<storage, read> sin_buf: array<f16>;

@group(0) @binding(4)
var<storage, read> rope_i_params: RopeIParams;

@compute @workgroup_size(32)
fn rope_i_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bh = rope_i_params.bh;
    let td = rope_i_params.td;
    let stride_b = rope_i_params.stride_b;
    let idx = gid.x;
    if (2u * idx >= bh * td) {
        return;
    }

    var rope_idx = idx % (td / 2u);
    if (stride_b > 0u) {
        let b_idx = (2u * idx) / stride_b;
        rope_idx += b_idx * (td / 2u);
    }
    let c = f32(cos_buf[rope_idx]);
    let s = f32(sin_buf[rope_idx]);
    let i0 = 2u * idx;
    let i1 = i0 + 1u;
    let v0 = f32(src_buf[i0]);
    let v1 = f32(src_buf[i1]);
    output_buf[i0] = f16(v0 * c - v1 * s);
    output_buf[i1] = f16(v0 * s + v1 * c);
}
