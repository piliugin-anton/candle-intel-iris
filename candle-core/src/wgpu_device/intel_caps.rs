use super::adapter::{is_intel_adapter, pci_id, INTEL_VENDOR_ID};
use crate::DType;
use wgpu::{AdapterInfo, Device, DeviceType};

/// Default UMA auto-map threshold: 4 MiB.
pub const DEFAULT_UMA_AUTO_MAP_THRESHOLD: usize = 4 * 1024 * 1024;

/// Intel GPU generation bucket used for dtype and tuning policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntelGeneration {
    /// Ice Lake (Gen 11) — FP16/BF16 compute emulated; prefer F32 kernels.
    Gen11,
    /// Tiger Lake / Xe (Gen 12+) — native FP16 on many SKUs.
    Gen12Plus,
    /// Pre-Gen11 or unrecognized Intel device.
    Older,
    /// Non-Intel adapter.
    NonIntel,
}

/// Intel Iris–specific capabilities cached at device creation.
#[derive(Clone, Debug)]
pub struct IntelCaps {
    pub generation: IntelGeneration,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub elem_workgroup_size: u32,
    pub reduce_workgroup_size: u32,
    pub matmul_tile_size: u32,
    pub supports_shader_f16: bool,
    pub supports_shader_bf16: bool,
    pub uma_auto_map_threshold: usize,
    pub force_fp32_compute: Option<bool>,
    pub is_integrated: bool,
}

impl IntelCaps {
    pub fn from_adapter(
        info: &AdapterInfo,
        device: &Device,
        uma_auto_map_threshold: usize,
        force_fp32_compute: Option<bool>,
    ) -> Self {
        let generation = detect_generation(info);
        let subgroup_min = info.subgroup_min_size;
        let subgroup_max = info.subgroup_max_size;
        let elem_workgroup_size = if subgroup_min > 0 {
            subgroup_min
        } else {
            crate::wgsl::ELEM_WORKGROUP_SIZE
        };
        let reduce_workgroup_size = elem_workgroup_size;
        let features = device.features();
        let supports_shader_f16 = features.contains(wgpu::Features::SHADER_F16);
        // wgpu 29 does not expose a dedicated bf16 shader feature; infer from generation.
        let supports_shader_bf16 =
            supports_shader_f16 && matches!(detect_generation(info), IntelGeneration::Gen12Plus);

        Self {
            generation,
            subgroup_min,
            subgroup_max,
            elem_workgroup_size,
            reduce_workgroup_size,
            matmul_tile_size: crate::wgsl::MATMUL_WORKGROUP_SIZE,
            supports_shader_f16,
            supports_shader_bf16,
            uma_auto_map_threshold,
            force_fp32_compute,
            is_integrated: info.device_type == DeviceType::IntegratedGpu,
        }
    }

    /// Conservative defaults when no real adapter is available (e.g. tests).
    pub fn default_fallback() -> Self {
        Self {
            generation: IntelGeneration::NonIntel,
            subgroup_min: 0,
            subgroup_max: 0,
            elem_workgroup_size: crate::wgsl::ELEM_WORKGROUP_SIZE,
            reduce_workgroup_size: crate::wgsl::REDUCE_WORKGROUP_SIZE,
            matmul_tile_size: crate::wgsl::MATMUL_WORKGROUP_SIZE,
            supports_shader_f16: false,
            supports_shader_bf16: false,
            uma_auto_map_threshold: DEFAULT_UMA_AUTO_MAP_THRESHOLD,
            force_fp32_compute: None,
            is_integrated: false,
        }
    }

    pub fn should_auto_map(&self, info: &AdapterInfo, byte_len: usize) -> bool {
        is_intel_adapter(info) && self.is_integrated && byte_len <= self.uma_auto_map_threshold
    }

    pub fn effective_compute_dtype(&self, requested: DType) -> DType {
        let force_fp32 = self
            .force_fp32_compute
            .unwrap_or(self.generation == IntelGeneration::Gen11);
        if force_fp32 {
            match requested {
                DType::F16 | DType::BF16 => DType::F32,
                other => other,
            }
        } else {
            requested
        }
    }

    pub fn supports_native_bf16(&self) -> bool {
        self.generation == IntelGeneration::Gen12Plus
    }

    pub fn supports_native_f16(&self) -> bool {
        self.generation == IntelGeneration::Gen12Plus && self.supports_shader_f16
    }
}

/// Detect Intel GPU generation from PCI device id and adapter name.
pub fn detect_generation(info: &AdapterInfo) -> IntelGeneration {
    if pci_id(info.vendor) != INTEL_VENDOR_ID {
        return IntelGeneration::NonIntel;
    }

    let device = pci_id(info.device);
    let name_lower = info.name.to_lowercase();

    if device >= 0x9A00
        || name_lower.contains("xe")
        || name_lower.contains("arc")
        || name_lower.contains("tiger")
    {
        return IntelGeneration::Gen12Plus;
    }

    if (0x8A00..=0x8FFF).contains(&device) || name_lower.contains("ice") {
        return IntelGeneration::Gen11;
    }

    IntelGeneration::Older
}

/// Replace a WGSL workgroup-size constant before shader compilation.
pub fn inject_workgroup_size(source: &str, const_name: &str, default: u32, size: u32) -> String {
    let needle = format!("const {const_name}: u32 = {default}u");
    let replacement = format!("const {const_name}: u32 = {size}u");
    source.replace(&needle, &replacement)
}

/// Inject element-wise and reduction workgroup sizes from device caps.
pub fn tune_shader_source(source: &str, caps: &IntelCaps) -> String {
    let mut out = inject_workgroup_size(
        source,
        "WG_SIZE",
        crate::wgsl::ELEM_WORKGROUP_SIZE,
        caps.elem_workgroup_size,
    );
    out = inject_workgroup_size(
        &out,
        "REDUCE_WG_SIZE",
        crate::wgsl::REDUCE_WORKGROUP_SIZE,
        caps.reduce_workgroup_size,
    );
    out
}

/// Inject matrix-multiply tile size from device caps.
pub fn tune_matmul_shader_source(source: &str, caps: &IntelCaps) -> String {
    inject_workgroup_size(
        source,
        "MATMUL_WG_SIZE",
        crate::wgsl::MATMUL_WORKGROUP_SIZE,
        caps.matmul_tile_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::DeviceType;

    fn intel_info(device: u32, name: &str) -> AdapterInfo {
        AdapterInfo {
            name: name.into(),
            vendor: INTEL_VENDOR_ID,
            device,
            device_type: DeviceType::IntegratedGpu,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 8,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn detect_gen11_ice_lake() {
        assert_eq!(
            detect_generation(&intel_info(0x8A52, "Intel Iris Plus Graphics")),
            IntelGeneration::Gen11
        );
    }

    #[test]
    fn detect_gen12_tiger_lake() {
        assert_eq!(
            detect_generation(&intel_info(0x9A49, "Intel Iris Xe Graphics")),
            IntelGeneration::Gen12Plus
        );
    }

    #[test]
    fn gen11_forces_fp32_compute() {
        let caps = IntelCaps {
            generation: IntelGeneration::Gen11,
            force_fp32_compute: None,
            ..IntelCaps::default_fallback()
        };
        assert_eq!(caps.effective_compute_dtype(DType::F16), DType::F32);
        assert_eq!(caps.effective_compute_dtype(DType::BF16), DType::F32);
        assert_eq!(caps.effective_compute_dtype(DType::F32), DType::F32);
    }

    #[test]
    fn gen12_preserves_half_dtypes() {
        let caps = IntelCaps {
            generation: IntelGeneration::Gen12Plus,
            force_fp32_compute: None,
            ..IntelCaps::default_fallback()
        };
        assert_eq!(caps.effective_compute_dtype(DType::F16), DType::F16);
        assert_eq!(caps.effective_compute_dtype(DType::BF16), DType::BF16);
    }

    #[test]
    fn tune_matmul_injects_tile_size() {
        let caps = IntelCaps {
            matmul_tile_size: 8,
            ..IntelCaps::default_fallback()
        };
        let src = "const MATMUL_WG_SIZE: u32 = 16u;";
        let out = tune_matmul_shader_source(src, &caps);
        assert!(out.contains("const MATMUL_WG_SIZE: u32 = 8u"));
    }

    #[test]
    fn inject_workgroup_size_replaces_constant() {
        let src = "const WG_SIZE: u32 = 32u;\n@workgroup_size(WG_SIZE)";
        let out = inject_workgroup_size(src, "WG_SIZE", 32, 8);
        assert!(out.contains("const WG_SIZE: u32 = 8u"));
        assert!(!out.contains("const WG_SIZE: u32 = 32u"));
    }

    #[test]
    fn auto_map_heuristic() {
        let caps = IntelCaps {
            uma_auto_map_threshold: 1024,
            is_integrated: true,
            ..IntelCaps::default_fallback()
        };
        let info = intel_info(0x9A49, "Intel Iris Xe");
        assert!(caps.should_auto_map(&info, 512));
        assert!(!caps.should_auto_map(&info, 2048));
    }
}
