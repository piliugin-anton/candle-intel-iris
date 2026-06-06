//! Embedded WGSL compute shaders for the wgpu backend.
//!
//! Shader modules are split into composable fragments under `candle-core/wgsl/`.
//! Rust concatenates `common` + operation-specific sources before compilation.

/// Default workgroup width before adapter-specific tuning at compile time.
pub const ELEM_WORKGROUP_SIZE: u32 = 32;

/// Workgroup width/height for matrix multiply kernels.
pub const MATMUL_WORKGROUP_SIZE: u32 = 16;

/// Workgroup width for reduction kernels.
pub const REDUCE_WORKGROUP_SIZE: u32 = 32;

pub const COMMON: &str = include_str!("../wgsl/common.wgsl");
pub const UNARY: &str = concat!(
    include_str!("../wgsl/common.wgsl"),
    include_str!("../wgsl/unary.wgsl")
);
pub const BINARY: &str = concat!(
    include_str!("../wgsl/common.wgsl"),
    include_str!("../wgsl/binary.wgsl")
);
pub const COPY: &str = concat!(
    include_str!("../wgsl/common.wgsl"),
    include_str!("../wgsl/copy.wgsl")
);
pub const COPY_F16: &str = concat!(
    include_str!("../wgsl/common_f16.wgsl"),
    include_str!("../wgsl/copy_f16.wgsl")
);
pub const COPY_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/copy_bf16.wgsl")
);
pub const REDUCE: &str = concat!(
    include_str!("../wgsl/reduce_common.wgsl"),
    include_str!("../wgsl/reduce.wgsl")
);
pub const REDUCE_F16: &str = concat!(
    include_str!("../wgsl/reduce_common_f16.wgsl"),
    include_str!("../wgsl/reduce_f16.wgsl")
);
pub const REDUCE_BF16: &str = concat!(
    include_str!("../wgsl/reduce_common_bf16.wgsl"),
    include_str!("../wgsl/reduce_bf16.wgsl")
);
pub const MATMUL_NAIVE: &str = concat!(
    include_str!("../wgsl/matmul_common.wgsl"),
    include_str!("../wgsl/matmul_naive.wgsl")
);
pub const MATMUL_TILED: &str = concat!(
    include_str!("../wgsl/matmul_common.wgsl"),
    include_str!("../wgsl/matmul_tiled.wgsl")
);
pub const MATMUL_TILED_VEC: &str = concat!(
    include_str!("../wgsl/matmul_common.wgsl"),
    include_str!("../wgsl/matmul_tiled_vec.wgsl")
);
pub const MATMUL_TILED_F16: &str = concat!(
    include_str!("../wgsl/matmul_common_f16.wgsl"),
    include_str!("../wgsl/matmul_tiled_f16.wgsl")
);
pub const MATMUL_TILED_BF16: &str = concat!(
    include_str!("../wgsl/matmul_common_bf16.wgsl"),
    include_str!("../wgsl/matmul_tiled_bf16.wgsl")
);
pub const QMATMUL_Q4_0: &str = concat!(
    include_str!("../wgsl/qmatmul_base.wgsl"),
    include_str!("../wgsl/qmatmul_q4_0_blocks.wgsl"),
    include_str!("../wgsl/qmatmul_q4_0.wgsl")
);
pub const QMATMUL_Q8_0: &str = concat!(
    include_str!("../wgsl/qmatmul_base.wgsl"),
    include_str!("../wgsl/qmatmul_q8_0_blocks.wgsl"),
    include_str!("../wgsl/qmatmul_q8_0.wgsl")
);
pub const QMATMUL_Q4_K: &str = concat!(
    include_str!("../wgsl/qmatmul_base.wgsl"),
    include_str!("../wgsl/qmatmul_q4k_blocks.wgsl"),
    include_str!("../wgsl/qmatmul_q4_k.wgsl")
);
pub const SOFTMAX: &str = include_str!("../wgsl/softmax.wgsl");
pub const SDPA_VECTOR: &str = include_str!("../wgsl/sdpa_vector.wgsl");
pub const SDPA_FULL: &str = include_str!("../wgsl/sdpa_full.wgsl");
pub const UNARY_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/unary_bf16.wgsl")
);
pub const BINARY_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/binary_bf16.wgsl")
);
pub const UNARY_F16: &str = concat!(
    include_str!("../wgsl/common_f16.wgsl"),
    include_str!("../wgsl/unary_f16.wgsl")
);
pub const BINARY_F16: &str = concat!(
    include_str!("../wgsl/common_f16.wgsl"),
    include_str!("../wgsl/binary_f16.wgsl")
);
pub const QMATMUL_Q5_0: &str = concat!(
    include_str!("../wgsl/qmatmul_base.wgsl"),
    include_str!("../wgsl/qmatmul_q5_0_blocks.wgsl"),
    include_str!("../wgsl/qmatmul_q5_0.wgsl")
);
pub const DEQUANT_Q4_0: &str = concat!(
    include_str!("../wgsl/dequant_base.wgsl"),
    include_str!("../wgsl/dequant_q4_0.wgsl")
);
pub const DEQUANT_Q5_0: &str = concat!(
    include_str!("../wgsl/dequant_base.wgsl"),
    include_str!("../wgsl/dequant_q5_0.wgsl")
);
pub const DEQUANT_Q8_0: &str = concat!(
    include_str!("../wgsl/dequant_base.wgsl"),
    include_str!("../wgsl/dequant_q8_0.wgsl")
);
pub const DEQUANT_Q4_K: &str = concat!(
    include_str!("../wgsl/dequant_base.wgsl"),
    include_str!("../wgsl/dequant_q4_k.wgsl")
);
pub const QUANT_Q4_0: &str = concat!(
    include_str!("../wgsl/quant_base.wgsl"),
    include_str!("../wgsl/quant_q4_0.wgsl")
);
pub const QUANT_Q5_0: &str = concat!(
    include_str!("../wgsl/quant_base.wgsl"),
    include_str!("../wgsl/quant_q5_0.wgsl")
);
pub const QUANT_Q8_0: &str = concat!(
    include_str!("../wgsl/quant_base.wgsl"),
    include_str!("../wgsl/quant_q8_0.wgsl")
);

/// Inner-loop vector width for tiled matmul kernels.
pub const MATMUL_VEC_WIDTH: u32 = 4;

pub const CAST: &str = include_str!("../wgsl/cast.wgsl");
pub const COPY2D: &str = include_str!("../wgsl/copy2d.wgsl");
pub const COPY2D_F16: &str = include_str!("../wgsl/copy2d_f16.wgsl");
pub const COPY2D_BF16: &str = include_str!("../wgsl/copy2d_bf16.wgsl");
pub const RMS_NORM: &str = include_str!("../wgsl/rms_norm.wgsl");
pub const ROPE: &str = include_str!("../wgsl/rope.wgsl");
pub const WHERE_COND: &str = include_str!("../wgsl/where_cond.wgsl");
pub const IM2COL2D: &str = include_str!("../wgsl/im2col2d.wgsl");
pub const IM2COL2D_F16: &str = include_str!("../wgsl/im2col2d_f16.wgsl");
pub const IM2COL2D_BF16: &str = include_str!("../wgsl/im2col2d_bf16.wgsl");
pub const IM2COL1D: &str = include_str!("../wgsl/im2col1d.wgsl");
pub const IM2COL1D_F16: &str = include_str!("../wgsl/im2col1d_f16.wgsl");
pub const IM2COL1D_BF16: &str = include_str!("../wgsl/im2col1d_bf16.wgsl");
pub const POOL2D: &str = include_str!("../wgsl/pool2d.wgsl");
pub const POOL2D_F16: &str = include_str!("../wgsl/pool2d_f16.wgsl");
pub const POOL2D_BF16: &str = include_str!("../wgsl/pool2d_bf16.wgsl");
pub const UPSAMPLE_NEAREST1D: &str = include_str!("../wgsl/upsample_nearest1d.wgsl");
pub const UPSAMPLE_NEAREST1D_F16: &str = include_str!("../wgsl/upsample_nearest1d_f16.wgsl");
pub const UPSAMPLE_NEAREST1D_BF16: &str = include_str!("../wgsl/upsample_nearest1d_bf16.wgsl");
pub const UPSAMPLE_NEAREST2D: &str = include_str!("../wgsl/upsample_nearest2d.wgsl");
pub const UPSAMPLE_NEAREST2D_F16: &str = include_str!("../wgsl/upsample_nearest2d_f16.wgsl");
pub const UPSAMPLE_NEAREST2D_BF16: &str = include_str!("../wgsl/upsample_nearest2d_bf16.wgsl");
pub const UPSAMPLE_BILINEAR2D: &str = include_str!("../wgsl/upsample_bilinear2d.wgsl");
pub const UPSAMPLE_BILINEAR2D_F16: &str = include_str!("../wgsl/upsample_bilinear2d_f16.wgsl");
pub const UPSAMPLE_BILINEAR2D_BF16: &str = include_str!("../wgsl/upsample_bilinear2d_bf16.wgsl");
pub const CONV_TRANSPOSE2D: &str = include_str!("../wgsl/conv_transpose2d.wgsl");
pub const CONV_TRANSPOSE2D_F16: &str = include_str!("../wgsl/conv_transpose2d_f16.wgsl");
pub const CONV_TRANSPOSE2D_BF16: &str = include_str!("../wgsl/conv_transpose2d_bf16.wgsl");
pub const CONV_TRANSPOSE1D: &str = include_str!("../wgsl/conv_transpose1d.wgsl");
pub const CONV_TRANSPOSE1D_F16: &str = include_str!("../wgsl/conv_transpose1d_f16.wgsl");
pub const CONV_TRANSPOSE1D_BF16: &str = include_str!("../wgsl/conv_transpose1d_bf16.wgsl");
pub const INDEXING: &str = include_str!("../wgsl/indexing.wgsl");
pub const CONST_SET_F32: &str = concat!(
    include_str!("../wgsl/common.wgsl"),
    include_str!("../wgsl/const_set_f32.wgsl")
);
pub const CONST_SET_F16: &str = concat!(
    include_str!("../wgsl/common_f16.wgsl"),
    include_str!("../wgsl/const_set_f16.wgsl")
);
pub const CONST_SET_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/const_set_bf16.wgsl")
);
pub const CONST_SET_U32: &str = include_str!("../wgsl/const_set_u32.wgsl");
pub const CONST_SET_U8: &str = include_str!("../wgsl/const_set_u8.wgsl");
pub const CMP: &str = include_str!("../wgsl/cmp.wgsl");
pub const RANDOM: &str = include_str!("../wgsl/random.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_sources_contain_entry_points() {
        assert!(UNARY.contains("fn neg_f32"));
        assert!(UNARY.contains("fn gelu_f32"));
        assert!(UNARY.contains("fn affine_f32"));
        assert!(BINARY.contains("fn add_f32"));
        assert!(COPY.contains("fn copy_strided_f32"));
        assert!(COPY_F16.contains("fn copy_strided_f16"));
        assert!(COPY_BF16.contains("fn copy_strided_bf16"));
        assert!(REDUCE.contains("fn reduce_sum_f32"));
        assert!(REDUCE_F16.contains("fn reduce_sum_f16"));
        assert!(REDUCE_BF16.contains("fn reduce_sum_bf16"));
        assert!(REDUCE.contains("fn reduce_min_f32"));
        assert!(MATMUL_NAIVE.contains("fn matmul_naive_f32"));
        assert!(MATMUL_TILED.contains("fn matmul_tiled_f32"));
        assert!(MATMUL_TILED_VEC.contains("fn matmul_tiled_vec_f32"));
        assert!(MATMUL_TILED_F16.contains("fn matmul_tiled_f16"));
        assert!(MATMUL_TILED_F16.contains("fn matmul_tiled_vec_f16"));
        assert!(QMATMUL_Q4_0.contains("fn qmatmul_q4_0_f32"));
        assert!(QMATMUL_Q5_0.contains("fn qmatmul_q5_0_f32"));
        assert!(UNARY_F16.contains("fn gelu_f16"));
        assert!(BINARY_F16.contains("fn add_f16"));
        assert!(QMATMUL_Q8_0.contains("fn qmatmul_q8_0_f32"));
        assert!(QMATMUL_Q4_K.contains("fn qmatmul_q4_k_f32"));
        assert!(SOFTMAX.contains("fn softmax_last_dim_f32"));
        assert!(SDPA_VECTOR.contains("fn sdpa_vector_f32"));
        assert!(SDPA_FULL.contains("fn sdpa_full_f32"));
        assert!(MATMUL_TILED_BF16.contains("fn matmul_tiled_bf16"));
        assert!(UNARY_BF16.contains("fn gelu_bf16"));
        assert!(UNARY_BF16.contains("fn silu_bf16"));
        assert!(UNARY_BF16.contains("fn affine_bf16"));
        assert!(BINARY_BF16.contains("fn add_bf16"));
        assert!(BINARY_BF16.contains("fn mul_bf16"));
        assert!(BINARY_BF16.contains("fn min_bf16"));
        assert!(CAST.contains("fn cast_f16_f32"));
        assert!(CAST.contains("fn f32_from_bf16_bits"));
        assert!(DEQUANT_Q4_0.contains("fn dequant_q4_0_f32"));
        assert!(DEQUANT_Q5_0.contains("fn dequant_q5_0_f32"));
        assert!(DEQUANT_Q8_0.contains("fn dequant_q8_0_f32"));
        assert!(DEQUANT_Q4_K.contains("fn dequant_q4_k_f32"));
        assert!(QUANT_Q4_0.contains("fn quant_q4_0_f32"));
        assert!(QUANT_Q5_0.contains("fn quant_q5_0_f32"));
        assert!(QUANT_Q8_0.contains("fn quant_q8_0_f32"));
        assert!(IM2COL2D.contains("fn im2col2d_f32"));
        assert!(IM2COL2D_F16.contains("fn im2col2d_f16"));
        assert!(IM2COL2D_BF16.contains("fn im2col2d_bf16"));
        assert!(IM2COL1D.contains("fn im2col1d_f32"));
        assert!(POOL2D.contains("fn avg_pool2d_f32"));
        assert!(POOL2D_BF16.contains("fn avg_pool2d_bf16"));
        assert!(COPY2D_F16.contains("fn copy2d_f16"));
        assert!(COPY2D_BF16.contains("fn copy2d_bf16"));
        assert!(POOL2D.contains("fn max_pool2d_f32"));
        assert!(UPSAMPLE_NEAREST1D.contains("fn upsample_nearest1d_f32"));
        assert!(UPSAMPLE_NEAREST2D.contains("fn upsample_nearest2d_f32"));
        assert!(UPSAMPLE_BILINEAR2D.contains("fn upsample_bilinear2d_f32"));
        assert!(CONV_TRANSPOSE2D.contains("fn conv_transpose2d_f32"));
        assert!(CONV_TRANSPOSE1D.contains("fn conv_transpose1d_f32"));
        assert!(INDEXING.contains("fn index_select_f32_u32"));
        assert!(INDEXING.contains("fn gather_f32_u32"));
        assert!(INDEXING.contains("fn scatter_f32_u32"));
        assert!(INDEXING.contains("fn index_add_f32_u32"));
        assert!(CONST_SET_F32.contains("fn const_set_f32"));
        assert!(CONST_SET_U32.contains("fn const_set_u32"));
        assert!(CONST_SET_U8.contains("fn const_set_u8"));
        assert!(CMP.contains("fn eq_f32"));
        assert!(CMP.contains("fn ge_f32"));
        assert!(UNARY.contains("fn powf_f32"));
        assert!(UNARY.contains("fn elu_f32"));
        assert!(UNARY_F16.contains("fn powf_f16"));
        assert!(UNARY_BF16.contains("fn powf_bf16"));
        assert!(RANDOM.contains("fn rand_uniform_f32"));
        assert!(RANDOM.contains("fn rand_normal_f32"));
    }

    #[test]
    fn common_defines_strided_index() {
        assert!(COMMON.contains("fn get_strided_index"));
    }

    #[test]
    fn elemwise_uses_tuned_workgroup_size() {
        assert!(UNARY.contains("@workgroup_size(WG_SIZE)"));
        assert!(BINARY.contains("@workgroup_size(WG_SIZE)"));
        assert!(!UNARY.contains("@workgroup_size(1)"));
        assert!(!BINARY.contains("@workgroup_size(1)"));
        assert!(UNARY.contains("grid_stride_x"));
        assert!(COMMON.contains("fn buffer_index"));
    }
}
