enable f16;
// NeoX-style rotary positional embedding (f16).
//
// Entry point: rope_f16

struct RopeParams {
    b: u32,
    h: u32,
    t: u32,
    d: u32,
    unbatched_cs: u32,
    _pad: array<u32, 67>,
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
var<storage, read> rope_params: RopeParams;

@compute @workgroup_size(32)
fn rope_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = rope_params.b;
    let h = rope_params.h;
    let t = rope_params.t;
    let d = rope_params.d;
    let half_d = d / 2u;
    let bh = b * h;
    let flat_bh = gid.x / t;
    let i_t = gid.x % t;
    if (flat_bh >= bh) {
        return;
    }

    let bh_i = flat_bh;
    let row_base = bh_i * t * d + i_t * d;
    for (var i_d = 0u; i_d < half_d; i_d = i_d + 1u) {
        let i1 = row_base + i_d;
        let i2 = i1 + half_d;
        var i_cs = i_t * half_d + i_d;
        if (rope_params.unbatched_cs == 0u) {
            let b_i = bh_i / h;
            i_cs += b_i * t * half_d;
        }
        let c = f32(cos_buf[i_cs]);
        let s = f32(sin_buf[i_cs]);
        let v1 = f32(src_buf[i1]);
        let v2 = f32(src_buf[i2]);
        output_buf[i1] = f16(v1 * c - v2 * s);
        output_buf[i2] = f16(v1 * s + v2 * c);
    }
}
