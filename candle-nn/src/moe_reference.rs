use candle::{quantized::QTensor, DType, Result, Tensor};

/// Reference MoE GEMM using generic tensor ops (works on wgpu and CPU).
pub fn moe_gemm_reference(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    let (input_rows, size_k) = input.dims2()?;
    let (_num_experts, size_n, size_k_w) = weights.dims3()?;
    if size_k != size_k_w {
        candle::bail!("input and weight last dim mismatch: {size_k} vs {size_k_w}");
    }

    let sorted = sorted_token_ids.flatten_all()?.to_vec1::<u32>()?;
    let experts = expert_ids.flatten_all()?.to_vec1::<u32>()?;
    let num_slots = sorted.len();
    if num_slots == 0 {
        let out_rows = if topk_weights.is_some() {
            input_rows
        } else {
            input_rows.saturating_mul(topk)
        };
        return Tensor::zeros((out_rows, size_n), input.dtype(), input.device());
    }

    let topk_eff = if topk_weights.is_some() {
        topk
    } else if input_rows > 0 {
        num_slots / input_rows
    } else {
        topk
    };
    if topk_eff == 0 {
        candle::bail!("moe_gemm invalid topk");
    }

    let mut rows = Vec::with_capacity(num_slots);
    for i in 0..num_slots {
        let flat_idx = sorted[i] as usize;
        let in_row = if input_rows == num_slots {
            flat_idx
        } else {
            flat_idx / topk_eff
        };
        let expert = experts[i] as usize;
        let in_slice = input.narrow(0, in_row, 1)?;
        let w = weights.narrow(0, expert, 1)?.squeeze(0)?;
        let mut row = in_slice.matmul(&w.t()?)?;
        if let Some(tw) = topk_weights {
            let scale = tw.flatten_all()?.narrow(0, flat_idx, 1)?;
            row = row.broadcast_mul(&scale)?;
        }
        rows.push(row);
    }
    Tensor::cat(&rows, 0)
}

/// Reference quantized MoE GEMM: dequantize expert weights then run the dense reference path.
pub fn moe_gemm_gguf_reference(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: DType,
) -> Result<Tensor> {
    let device = input.device();
    let dense = if is_prefill {
        weights.dequantize_f16(device)?.to_dtype(dtype)?
    } else {
        weights.dequantize(device)?
    };
    moe_gemm_reference(
        input,
        &dense,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        topk,
    )
}
