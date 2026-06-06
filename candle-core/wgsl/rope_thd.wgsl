// THD-layout rotary positional embedding (f32): tensor shape (b, t, h, d).
//
// Entry point: rope_thd_f32

struct RopeThdParams {
    b: u32,
    t: u32,
    h: u32,
    d: u32,
    stride_b: u32,
    _pad: array<u32, 67>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> src_buf: array<f32>;

@group(0) @binding(2)
var<storage, read> cos_buf: array<f32>;

@group(0) @binding(3)
var<storage, read> sin_buf: array<f32>;

@group(0) @binding(4)
var<storage, read> rope_thd_params: RopeThdParams;

@compute @workgroup_size(32)
fn rope_thd_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = rope_thd_params.b;
    let t = rope_thd_params.t;
    let h = rope_thd_params.h;
    let d = rope_thd_params.d;
    let stride_b = rope_thd_params.stride_b;
    let idx = gid.x;
    if (2u * idx >= b * t * h * d) {
        return;
    }

    let i_bth = idx / (d / 2u);
    let i_d = idx - (d / 2u) * i_bth;
    let i_t = (i_bth / h) % t;
    let i1 = i_bth * d + i_d;
    let i2 = i1 + d / 2u;
    var i_cs = i_t * (d / 2u) + i_d;
    if (stride_b > 0u) {
        let b_idx = (2u * idx) / stride_b;
        i_cs += b_idx * ((t * d) / 2u);
    }
    let c = cos_buf[i_cs];
    let s = sin_buf[i_cs];
    let v1 = src_buf[i1];
    let v2 = src_buf[i2];
    output_buf[i1] = v1 * c - v2 * s;
    output_buf[i2] = v1 * s + v2 * c;
}
