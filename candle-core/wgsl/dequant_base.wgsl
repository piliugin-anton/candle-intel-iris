// Shared bindings and byte helpers for GPU dequantization kernels.

struct DequantParams {
    elem_count: u32,
    _pad: array<u32, 71>,
}

@group(0) @binding(0)
var<storage, read_write> dequant_out: array<f32>;

@group(0) @binding(1)
var<storage, read> dequant_in: array<u32>;

// Binding 2 is unused; kept for the standard 4-slot Candle kernel layout.
@group(0) @binding(2)
var<storage, read> dequant_in_pad: array<u32>;

@group(0) @binding(3)
var<storage, read> dequant_params: DequantParams;

fn dequant_read_byte(byte_idx: u32) -> u32 {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (dequant_in[word] >> shift) & 0xFFu;
}

fn f16_bytes_to_f32(lo: u32, hi: u32) -> f32 {
    return f32(unpack2x16float(lo | (hi << 8u)).x);
}
