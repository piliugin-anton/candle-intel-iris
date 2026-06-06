// Shared bindings and byte helpers for GPU quantization kernels.

struct QuantParams {
    elem_count: u32,
    _pad: array<u32, 71>,
}

@group(0) @binding(0)
var<storage, read_write> quant_out: array<u32>;

@group(0) @binding(1)
var<storage, read> quant_src: array<f32>;

// Binding 2 is unused; kept for the standard 4-slot Candle kernel layout.
@group(0) @binding(2)
var<storage, read> quant_src_pad: array<f32>;

@group(0) @binding(3)
var<storage, read> quant_params: QuantParams;

fn quant_read_byte(byte_idx: u32) -> u32 {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (quant_out[word] >> shift) & 0xFFu;
}

fn quant_write_byte(byte_idx: u32, value: u32) {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    let mask = ~(0xFFu << shift);
    let old = quant_out[word];
    quant_out[word] = (old & mask) | ((value & 0xFFu) << shift);
}

fn f32_to_f16_bytes(value: f32) -> vec2<u32> {
    let bits = pack2x16float(vec2<f32>(value, 0.0));
    return vec2<u32>(bits & 0xFFu, (bits >> 8u) & 0xFFu);
}
