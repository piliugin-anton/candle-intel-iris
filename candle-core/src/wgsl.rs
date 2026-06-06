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
pub const CAST: &str = include_str!("../wgsl/cast.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_sources_contain_entry_points() {
        assert!(UNARY.contains("fn neg_f32"));
        assert!(BINARY.contains("fn add_f32"));
        assert!(COPY.contains("fn copy_strided_f32"));
        assert!(REDUCE.contains("fn reduce_sum_f32"));
        assert!(MATMUL_NAIVE.contains("fn matmul_naive_f32"));
        assert!(MATMUL_TILED.contains("fn matmul_tiled_f32"));
        assert!(CAST.contains("fn cast_f16_f32"));
    }

    #[test]
    fn common_defines_strided_index() {
        assert!(COMMON.contains("fn get_strided_index"));
    }
}
