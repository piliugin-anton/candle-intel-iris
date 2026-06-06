// Dtype cast kernels (storage reinterpreted via u32 words).
//
// Entry points: cast_f16_f32, cast_f32_f16, cast_bf16_f32, cast_f32_bf16,
//               cast_f16_bf16, cast_bf16_f16, cast_u8_f32, cast_f32_u8,
//               cast_f32_u32
//
// Packed dtypes (f16/bf16/u8) use one thread per u32 word so writes never race.

const MAX_DIMS: u32 = 8u;
const CAST_WG_SIZE: u32 = 32u;

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
var<storage, read_write> out_words: array<u32>;

@group(0) @binding(1)
var<storage, read> in0_words: array<u32>;

@group(0) @binding(2)
var<storage, read> in1_words: array<u32>;

@group(0) @binding(3)
var<storage, read> kernel_params: KernelParams;

fn cast_stride(num_wg: vec3<u32>) -> u32 {
    return CAST_WG_SIZE * num_wg.x;
}

fn read_f16_elem(elem_idx: u32) -> f32 {
    let word = elem_idx / 2u;
    let pair = unpack2x16float(in0_words[word]);
    return select(pair.x, pair.y, (elem_idx % 2u) == 1u);
}

fn write_f16_elem(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    var pair = unpack2x16float(out_words[word]);
    if (elem_idx % 2u) == 0u {
        pair = vec2<f32>(value, pair.y);
    } else {
        pair = vec2<f32>(pair.x, value);
    }
    out_words[word] = pack2x16float(pair);
}

fn f32_from_bf16_bits(bf16: u32) -> f32 {
    return bitcast<f32>(bf16 << 16u);
}

fn bf16_bits_from_f32(value: f32) -> u32 {
    return (bitcast<u32>(value) >> 16u) & 0xFFFFu;
}

fn read_bf16_elem(elem_idx: u32) -> f32 {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let packed = in0_words[word];
    let bf16 = (packed >> (byte_off * 8u)) & 0xFFFFu;
    return f32_from_bf16_bits(bf16);
}

fn write_bf16_elem(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    let byte_off = (elem_idx % 2u) * 2u;
    let shift = byte_off * 8u;
    let bf16 = bf16_bits_from_f32(value);
    let mask = ~(0xFFFFu << shift);
    var packed = out_words[word];
    packed = (packed & mask) | (bf16 << shift);
    out_words[word] = packed;
}

fn read_f32_elem(elem_idx: u32) -> f32 {
    return bitcast<f32>(in0_words[elem_idx]);
}

fn write_f32_elem(elem_idx: u32, value: f32) {
    out_words[elem_idx] = bitcast<u32>(value);
}

fn read_u8_elem(elem_idx: u32) -> f32 {
    let word = elem_idx / 4u;
    let byte = elem_idx % 4u;
    let packed = in0_words[word];
    let shift = byte * 8u;
    return f32((packed >> shift) & 0xFFu);
}

fn write_u8_elem(elem_idx: u32, value: f32) {
    let word = elem_idx / 4u;
    let byte = elem_idx % 4u;
    let shift = byte * 8u;
    let mask = ~(0xFFu << shift);
    var packed = out_words[word];
    packed = (packed & mask) | (u32(value) << shift);
    out_words[word] = packed;
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f16_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let pair = unpack2x16float(in0_words[w]);
        let i0 = w * 2u;
        write_f32_elem(i0, pair.x);
        if (i0 + 1u < count) {
            write_f32_elem(i0 + 1u, pair.y);
        }
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f32_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let i0 = w * 2u;
        var pair = vec2<f32>(0.0, 0.0);
        if (i0 < count) {
            pair.x = read_f32_elem(i0);
        }
        if (i0 + 1u < count) {
            pair.y = read_f32_elem(i0 + 1u);
        }
        out_words[w] = pack2x16float(pair);
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_bf16_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let packed = in0_words[w];
        let i0 = w * 2u;
        write_f32_elem(i0, f32_from_bf16_bits(packed & 0xFFFFu));
        if (i0 + 1u < count) {
            write_f32_elem(i0 + 1u, f32_from_bf16_bits((packed >> 16u) & 0xFFFFu));
        }
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f32_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let i0 = w * 2u;
        var lo = 0u;
        var hi = 0u;
        if (i0 < count) {
            lo = bf16_bits_from_f32(read_f32_elem(i0));
        }
        if (i0 + 1u < count) {
            hi = bf16_bits_from_f32(read_f32_elem(i0 + 1u));
        }
        out_words[w] = lo | (hi << 16u);
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f16_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let i0 = w * 2u;
        var lo = 0u;
        var hi = 0u;
        if (i0 < count) {
            lo = bf16_bits_from_f32(read_f16_elem(i0));
        }
        if (i0 + 1u < count) {
            hi = bf16_bits_from_f32(read_f16_elem(i0 + 1u));
        }
        out_words[w] = lo | (hi << 16u);
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_bf16_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 1u) / 2u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let packed = in0_words[w];
        let i0 = w * 2u;
        if (i0 < count) {
            write_f16_elem(i0, f32_from_bf16_bits(packed & 0xFFFFu));
        }
        if (i0 + 1u < count) {
            write_f16_elem(i0 + 1u, f32_from_bf16_bits((packed >> 16u) & 0xFFFFu));
        }
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_u8_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 3u) / 4u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        let packed = in0_words[w];
        for (var b = 0u; b < 4u; b = b + 1u) {
            let i = w * 4u + b;
            if (i < count) {
                write_f32_elem(i, f32((packed >> (b * 8u)) & 0xFFu));
            }
        }
    }
}

@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f32_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    let word_count = (count + 3u) / 4u;
    for (var w = gid.x; w < word_count; w = w + stride) {
        var packed = 0u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let i = w * 4u + b;
            if (i < count) {
                packed |= u32(read_f32_elem(i)) << (b * 8u);
            }
        }
        out_words[w] = packed;
    }
}

// Argmax/argmin kernels write indices as f32; Candle expects u32 storage.
@compute @workgroup_size(CAST_WG_SIZE)
fn cast_f32_u32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let stride = cast_stride(num_wg);
    let count = kernel_params.elem_count;
    for (var i = gid.x; i < count; i = i + stride) {
        out_words[i] = u32(read_f32_elem(i));
    }
}
