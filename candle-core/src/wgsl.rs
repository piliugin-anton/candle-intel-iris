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
pub const REDUCE: &str = concat!(
    include_str!("../wgsl/reduce_common.wgsl"),
    include_str!("../wgsl/reduce.wgsl")
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
pub const UNARY_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/unary_bf16.wgsl")
);
pub const BINARY_BF16: &str = concat!(
    include_str!("../wgsl/common_bf16.wgsl"),
    include_str!("../wgsl/binary_bf16.wgsl")
);

/// Inner-loop vector width for tiled matmul kernels.
pub const MATMUL_VEC_WIDTH: u32 = 4;

pub const CAST: &str = include_str!("../wgsl/cast.wgsl");
pub const COPY2D: &str = include_str!("../wgsl/copy2d.wgsl");
pub const RMS_NORM: &str = include_str!("../wgsl/rms_norm.wgsl");
pub const ROPE: &str = include_str!("../wgsl/rope.wgsl");
pub const WHERE_COND: &str = include_str!("../wgsl/where_cond.wgsl");

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
        assert!(REDUCE.contains("fn reduce_sum_f32"));
        assert!(REDUCE.contains("fn reduce_min_f32"));
        assert!(MATMUL_NAIVE.contains("fn matmul_naive_f32"));
        assert!(MATMUL_TILED.contains("fn matmul_tiled_f32"));
        assert!(MATMUL_TILED_VEC.contains("fn matmul_tiled_vec_f32"));
        assert!(MATMUL_TILED_F16.contains("fn matmul_tiled_f16"));
        assert!(MATMUL_TILED_F16.contains("fn matmul_tiled_vec_f16"));
        assert!(QMATMUL_Q4_0.contains("fn qmatmul_q4_0_f32"));
        assert!(QMATMUL_Q8_0.contains("fn qmatmul_q8_0_f32"));
        assert!(QMATMUL_Q4_K.contains("fn qmatmul_q4_k_f32"));
        assert!(SOFTMAX.contains("fn softmax_last_dim_f32"));
        assert!(SDPA_VECTOR.contains("fn sdpa_vector_f32"));
        assert!(MATMUL_TILED_BF16.contains("fn matmul_tiled_bf16"));
        assert!(UNARY_BF16.contains("fn gelu_bf16"));
        assert!(BINARY_BF16.contains("fn add_bf16"));
        assert!(CAST.contains("fn cast_f16_f32"));
        assert!(CAST.contains("fn f32_from_bf16_bits"));
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
