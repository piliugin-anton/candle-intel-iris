// Dtype cast kernels (storage reinterpreted via u32 words).
//
// Entry points: cast_f16_f32, cast_f32_f16, cast_bf16_f32, cast_f32_bf16,
//               cast_f16_bf16, cast_bf16_f16, cast_u8_f32, cast_f32_u8

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

fn read_bf16_elem(elem_idx: u32) -> f32 {
    let word = elem_idx / 2u;
    let pair = unpack2x16float(in0_words[word]);
    let v = select(pair.x, pair.y, (elem_idx % 2u) == 1u);
    return f32(v);
}

fn write_bf16_elem(elem_idx: u32, value: f32) {
    let word = elem_idx / 2u;
    var pair = unpack2x16float(out_words[word]);
    let half = f32(value);
    if (elem_idx % 2u) == 0u {
        pair = vec2<f32>(half, pair.y);
    } else {
        pair = vec2<f32>(pair.x, half);
    }
    out_words[word] = pack2x16float(pair);
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

// Packed dtypes (f16/bf16/u8) share u32 words — run serially in one workgroup to avoid write races.
@compute @workgroup_size(1)
fn cast_f16_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_f32_elem(i, read_f16_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_f32_f16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_f16_elem(i, read_f32_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_bf16_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_f32_elem(i, read_bf16_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_f32_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_bf16_elem(i, read_f32_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_f16_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_bf16_elem(i, read_f16_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_bf16_f16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_f16_elem(i, read_bf16_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_u8_f32(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_f32_elem(i, read_u8_elem(i));
    }
}

@compute @workgroup_size(1)
fn cast_f32_u8(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        write_u8_elem(i, read_f32_elem(i));
    }
}
