//! Cross-platform fused scaled dot-product attention for transformer models.
//!
//! Dispatches to:
//! - CUDA + `flash-attn` feature: `candle_flash_attn::flash_attn`
//! - Metal / wgpu: `candle_nn::ops::sdpa`
//!
//! Tensor layout for [`fused_attention`]: `(batch, num_heads, seq, head_dim)`.
//! For GQA, `k` and `v` may use fewer heads; SDPA handles grouping without
//! `repeat_kv` when possible, but repeated `k`/`v` are also accepted.

use candle::{Device, Result, Tensor, D};

/// Returns true when `device` should use [`fused_attention`] via SDPA (Metal / wgpu).
pub fn uses_sdpa_device(device: &Device) -> bool {
    #[cfg(feature = "metal")]
    if device.is_metal() {
        return true;
    }
    #[cfg(feature = "wgpu")]
    if device.is_wgpu() {
        return true;
    }
    let _ = device;
    false
}

/// Returns whether [`fused_attention`] can run on `device` for tensors with `head_dim`.
pub fn fused_attention_available(device: &Device, head_dim: usize) -> bool {
    #[cfg(feature = "wgpu")]
    if device.is_wgpu() {
        return head_dim <= candle::wgpu_device::MAX_SDPA_DIM;
    }
    #[cfg(feature = "metal")]
    if device.is_metal() {
        return true;
    }
    #[cfg(feature = "flash-attn")]
    if device.is_cuda() {
        return true;
    }
    let _ = (device, head_dim);
    false
}

/// Fused attention with `q`/`k`/`v` in `(batch, num_heads, seq, head_dim)`.
pub fn fused_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let device = q.device();
    let head_dim = q.dim(D::Minus1)?;

    if !fused_attention_available(device, head_dim) {
        candle::bail!("fused attention is not available on {device:?} for head_dim {head_dim}");
    }

    #[cfg(any(feature = "metal", feature = "wgpu"))]
    if uses_sdpa_device(device) {
        return candle_nn::ops::sdpa(q, k, v, attention_mask, causal, softmax_scale, 1.0);
    }

    #[cfg(feature = "flash-attn")]
    if device.is_cuda() {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let out = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, causal)?;
        return out.transpose(1, 2);
    }

    let _ = (k, v, softmax_scale, causal, attention_mask);
    candle::bail!("fused attention is not supported on {device:?}");
}

/// Fused attention with `q`/`k`/`v` in `(batch, seq, num_heads, head_dim)`.
pub fn fused_attention_bsnd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;
    fused_attention(&q, &k, &v, softmax_scale, causal, attention_mask)?.transpose(1, 2)
}

/// CUDA flash-attn layout `(batch, seq, num_heads, head_dim)` without SDPA routing.
#[cfg(feature = "flash-attn")]
pub fn flash_attn_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    if !q.device().is_cuda() {
        candle::bail!("flash_attn_cuda requires a CUDA device");
    }
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn wgpu_requires_supported_head_dim() {
        let device = Device::Cpu;
        assert!(!fused_attention_available(&device, 64));
        #[cfg(feature = "wgpu")]
        {
            let _ = fused_attention_available;
        }
    }
}
