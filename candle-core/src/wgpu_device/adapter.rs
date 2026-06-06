use super::error::{Result, WgpuError};
use super::intel_caps::{IntelCaps, DEFAULT_UMA_AUTO_MAP_THRESHOLD};
use super::WgpuDevice;
use wgpu::{
    Adapter, AdapterInfo, Backends, DeviceDescriptor, DeviceType, Instance, InstanceFlags,
    PowerPreference,
};

/// PCI vendor ID for Intel GPUs.
pub const INTEL_VENDOR_ID: u32 = 0x8086;

/// Configuration for wgpu device initialization and Intel adapter selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuDeviceConfig {
    /// Power preference used when no explicit [`device_id`](Self::device_id) is set.
    ///
    /// [`PowerPreference::LowPower`] selects the integrated Intel GPU (e.g. Iris).
    pub power_preference: PowerPreference,
    /// Explicit PCI device ID (lower 16 bits of [`AdapterInfo::device`]).
    pub device_id: Option<u32>,
    /// Byte threshold for transparent UMA mapped buffer allocation on Intel integrated GPUs.
    pub uma_auto_map_threshold: usize,
    /// When set, overrides automatic Gen11 FP32 compute policy.
    pub force_fp32_compute: Option<bool>,
    /// wgpu instance flags (`WGPU_VALIDATION=1` enables validation when using the default).
    pub instance_flags: InstanceFlags,
}

impl Default for WgpuDeviceConfig {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::LowPower,
            device_id: None,
            uma_auto_map_threshold: DEFAULT_UMA_AUTO_MAP_THRESHOLD,
            force_fp32_compute: None,
            instance_flags: InstanceFlags::from_env_or_default(),
        }
    }
}

impl WgpuDevice {
    /// Creates a device using the default Intel-focused configuration.
    pub fn new_default() -> Result<Self> {
        Self::new_with_config(WgpuDeviceConfig::default())
    }

    /// Creates a device after enumerating adapters and selecting an Intel GPU.
    pub fn new_with_config(config: WgpuDeviceConfig) -> Result<Self> {
        pollster::block_on(Self::new_with_config_async(config))
    }

    /// Async variant of [`Self::new_with_config`].
    pub async fn new_with_config_async(config: WgpuDeviceConfig) -> Result<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            flags: config.instance_flags,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = instance.enumerate_adapters(Backends::all()).await;
        let adapter = select_intel_adapter(adapters, &config)?;
        let info = adapter.get_info();

        let mut required_features = wgpu::Features::empty();
        let adapter_features = adapter.features();
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features,
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|err| WgpuError::Message(err.to_string()))?;

        let caps = IntelCaps::from_adapter(
            &info,
            &device,
            config.uma_auto_map_threshold,
            config.force_fp32_compute,
        );

        Ok(Self::from_parts(device, queue, info, caps))
    }

    pub(crate) fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: AdapterInfo,
        caps: IntelCaps,
    ) -> Self {
        Self {
            id: super::DeviceId::new(),
            device,
            queue,
            adapter_info,
            caps,
            shader_cache: super::ShaderCache::new(),
            allocator: super::Allocator::new(),
        }
    }
}

/// Returns true when `info` describes an Intel GPU.
pub fn is_intel_adapter(info: &AdapterInfo) -> bool {
    pci_id(info.vendor) == INTEL_VENDOR_ID
}

pub(crate) fn pci_id(raw: u32) -> u32 {
    raw & 0xFFFF
}

fn select_intel_adapter(adapters: Vec<Adapter>, config: &WgpuDeviceConfig) -> Result<Adapter> {
    let intel_adapters: Vec<(Adapter, AdapterInfo)> = adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            (adapter, info)
        })
        .filter(|(_, info)| is_intel_adapter(info))
        .collect();

    let index = select_intel_adapter_index(
        &intel_adapters
            .iter()
            .map(|(_, info)| info.clone())
            .collect::<Vec<_>>(),
        config,
    )?;
    let len = intel_adapters.len();
    intel_adapters
        .into_iter()
        .nth(index)
        .map(|(adapter, _)| adapter)
        .ok_or(WgpuError::AdapterIndexOutOfRange { index, len })
}

/// Picks an Intel adapter index from a pre-filtered list of [`AdapterInfo`].
fn select_intel_adapter_index(infos: &[AdapterInfo], config: &WgpuDeviceConfig) -> Result<usize> {
    if infos.is_empty() {
        return Err(WgpuError::NoIntelAdapter);
    }

    if let Some(device_id) = config.device_id {
        return infos
            .iter()
            .position(|info| pci_id(info.device) == device_id)
            .ok_or(WgpuError::IntelDeviceNotFound(device_id));
    }

    let preferred_type = match config.power_preference {
        PowerPreference::LowPower => Some(DeviceType::IntegratedGpu),
        PowerPreference::HighPerformance => Some(DeviceType::DiscreteGpu),
        PowerPreference::None => None,
    };

    if let Some(device_type) = preferred_type {
        if let Some(index) = infos
            .iter()
            .position(|info| info.device_type == device_type)
        {
            return Ok(index);
        }
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intel_adapter_info(device: u32, device_type: DeviceType) -> AdapterInfo {
        AdapterInfo {
            name: "Intel Test GPU".into(),
            vendor: INTEL_VENDOR_ID,
            device,
            device_type,
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
    fn is_intel_adapter_matches_vendor_id() {
        let info = intel_adapter_info(0x9a49, DeviceType::IntegratedGpu);
        assert!(is_intel_adapter(&info));

        let other = AdapterInfo {
            vendor: 0x10de,
            ..intel_adapter_info(0, DeviceType::DiscreteGpu)
        };
        assert!(!is_intel_adapter(&other));
    }

    #[test]
    fn select_by_device_id() {
        let integrated = intel_adapter_info(0x9a49, DeviceType::IntegratedGpu);
        let discrete = intel_adapter_info(0x1234, DeviceType::DiscreteGpu);
        let infos = vec![discrete.clone(), integrated.clone()];

        let index = select_intel_adapter_index(
            &infos,
            &WgpuDeviceConfig {
                device_id: Some(0x9a49),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(index, 1);
    }

    #[test]
    fn select_low_power_prefers_integrated() {
        let integrated = intel_adapter_info(0x9a49, DeviceType::IntegratedGpu);
        let discrete = intel_adapter_info(0x1234, DeviceType::DiscreteGpu);
        let infos = vec![discrete, integrated];

        let index = select_intel_adapter_index(
            &infos,
            &WgpuDeviceConfig {
                power_preference: PowerPreference::LowPower,
                device_id: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(index, 1);
    }

    #[test]
    fn default_config_prefers_low_power() {
        let config = WgpuDeviceConfig::default();
        assert_eq!(config.power_preference, PowerPreference::LowPower);
        assert!(config.device_id.is_none());
        assert_eq!(
            config.uma_auto_map_threshold,
            DEFAULT_UMA_AUTO_MAP_THRESHOLD
        );
        assert_eq!(config.instance_flags, InstanceFlags::from_env_or_default());
    }
}
