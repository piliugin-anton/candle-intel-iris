// 2D region copy for u8 tensors (byte-packed in u32 words, used by Tensor::cat).
//
// Parallel byte writes use compare-exchange on packed u32 words to avoid races when
// multiple threads update different bytes in the same word.
//
// Entry point: copy2d_u8

struct Copy2dParams {
    d1: u32,
    d2: u32,
    src_stride: u32,
    dst_stride: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: array<u32, 66>,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> input0_buf: array<u32>;

@group(0) @binding(3)
var<storage, read> copy2d_params: Copy2dParams;

fn load_u8(byte_idx: u32) -> u32 {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (input0_buf[word] >> shift) & 0xFFu;
}

fn store_u8(byte_idx: u32, value: u32) {
    let word = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    let mask = 0xFFu << shift;
    let new_byte = (value & 0xFFu) << shift;
    var old = atomicLoad(&output_buf[word]);
    loop {
        let new_val = (old & ~mask) | new_byte;
        let res = atomicCompareExchangeWeak(&output_buf[word], old, new_val);
        if (res.exchanged) {
            break;
        }
        old = res.old_value;
    }
}

fn copy_words_1d(src_base: u32, dst_base: u32, len: u32, thread_id: u32, stride: u32) {
    let words = (len + 3u) >> 2u;
    let src_word_base = src_base >> 2u;
    let dst_word_base = dst_base >> 2u;
    if (((src_base | dst_base | len) & 3u) == 0u) {
        for (var wi = thread_id; wi < words; wi = wi + stride) {
            atomicStore(&output_buf[dst_word_base + wi], input0_buf[src_word_base + wi]);
        }
    } else {
        for (var col = thread_id; col < len; col = col + stride) {
            store_u8(dst_base + col, load_u8(src_base + col));
        }
    }
}

@compute @workgroup_size(32)
fn copy2d_u8(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let d1 = copy2d_params.d1;
    let d2 = copy2d_params.d2;
    let src_stride = copy2d_params.src_stride;
    let dst_stride = copy2d_params.dst_stride;
    let src_base = copy2d_params.src_offset;
    let dst_base = copy2d_params.dst_offset;
    let stride = 32u * num_wg.x;

    if (d1 == 1u) {
        copy_words_1d(src_base, dst_base, d2, gid.x, stride);
    } else {
        let total = d1 * d2;
        for (var flat = gid.x; flat < total; flat = flat + stride) {
            let row = flat / d2;
            let col = flat % d2;
            let src_idx = src_base + row * src_stride + col;
            let dst_idx = dst_base + row * dst_stride + col;
            store_u8(dst_idx, load_u8(src_idx));
        }
    }
}
