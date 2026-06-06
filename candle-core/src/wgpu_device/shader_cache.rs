use super::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PipelineKey {
    source: Arc<str>,
    entry_point: Arc<str>,
    /// Distinguishes pipelines compiled with different bind group layouts.
    /// `0` means wgpu's implicit layout (`layout: None`).
    layout_key: u64,
}

/// Cache key for pipelines using [`super::bind_group::StandardBindGroupLayout`].
pub const STANDARD_KERNEL_LAYOUT_KEY: u64 = 1;

/// Cache for compiled WGSL compute pipelines.
///
/// Shader modules are kept alive alongside their pipelines so that pipeline
/// handles remain valid for the lifetime of the cache.
#[derive(Clone)]
pub struct ShaderCache {
    modules: Arc<RwLock<HashMap<Arc<str>, wgpu::ShaderModule>>>,
    pipelines: Arc<RwLock<HashMap<PipelineKey, wgpu::ComputePipeline>>>,
}

impl std::fmt::Debug for ShaderCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderCache").finish_non_exhaustive()
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderCache {
    pub fn new() -> Self {
        Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            pipelines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn get_or_create_pipeline(
        &self,
        device: &wgpu::Device,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline> {
        self.get_or_create_pipeline_with_layout(device, None, 0, source, entry_point)
    }

    pub fn get_or_create_pipeline_with_layout(
        &self,
        device: &wgpu::Device,
        bind_group_layout: Option<&wgpu::BindGroupLayout>,
        layout_key: u64,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let source: Arc<str> = Arc::from(source);
        let entry_point: Arc<str> = Arc::from(entry_point);
        let key = PipelineKey {
            source: source.clone(),
            entry_point: entry_point.clone(),
            layout_key,
        };

        {
            let pipelines = self.pipelines.read()?;
            if let Some(pipeline) = pipelines.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        let module = {
            let mut modules = self.modules.write()?;
            if let Some(module) = modules.get(&source) {
                module.clone()
            } else {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source.as_ref())),
                });
                modules.insert(source.clone(), module.clone());
                module
            }
        };

        let pipeline_layout = bind_group_layout.map(|layout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("candle wgpu kernel pipeline layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            })
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: pipeline_layout.as_ref(),
            module: &module,
            entry_point: Some(entry_point.as_ref()),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    pub fn clear(&self) -> Result<()> {
        self.pipelines.write()?.clear();
        self.modules.write()?.clear();
        Ok(())
    }
}
