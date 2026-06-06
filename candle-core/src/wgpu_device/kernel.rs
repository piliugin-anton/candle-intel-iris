use super::async_io::poll_device;
use super::bind_group::{
    BindGroupBuilder, ExtendedBindGroupArgs, ExtendedBindGroupBuilder, KernelUniforms,
    SdpaBindGroupArgs, SdpaBindGroupBuilder, StandardBindGroupArgs, StandardBindGroupLayout,
};
use super::shader_cache::{EXTENDED_KERNEL_LAYOUT_KEY, SDPA_KERNEL_LAYOUT_KEY};
use super::error::Result;
use super::intel_caps::tune_shader_source;
use super::shader_cache::{ShaderCache, STANDARD_KERNEL_LAYOUT_KEY};
use super::storage::BufferOffset;
use super::WgpuDevice;
use crate::Layout;

/// Holds a compiled compute pipeline and utilities for dispatching it.
#[derive(Clone, Debug)]
pub struct WgpuKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_builder: BindGroupBuilder,
    extended_bind_group_builder: Option<ExtendedBindGroupBuilder>,
    sdpa_bind_group_builder: Option<SdpaBindGroupBuilder>,
    workgroup_size: u32,
}

/// Tensor layouts and buffer bindings for a kernel launch.
pub struct KernelDispatchArgs<'a> {
    pub output: BufferOffset<'a>,
    pub input0: BufferOffset<'a>,
    pub input1: Option<BufferOffset<'a>>,
    pub out_layout: &'a Layout,
    pub in0_layout: &'a Layout,
    pub in1_layout: Option<&'a Layout>,
}

impl<'a> KernelDispatchArgs<'a> {
    pub fn unary(
        output: BufferOffset<'a>,
        input0: BufferOffset<'a>,
        out_layout: &'a Layout,
        in0_layout: &'a Layout,
    ) -> Self {
        Self {
            output,
            input0,
            input1: None,
            out_layout,
            in0_layout,
            in1_layout: None,
        }
    }

    pub fn binary(
        output: BufferOffset<'a>,
        input0: BufferOffset<'a>,
        input1: BufferOffset<'a>,
        out_layout: &'a Layout,
        in0_layout: &'a Layout,
        in1_layout: &'a Layout,
    ) -> Self {
        Self {
            output,
            input0,
            input1: Some(input1),
            out_layout,
            in0_layout,
            in1_layout: Some(in1_layout),
        }
    }

    fn elem_count(&self) -> usize {
        self.out_layout.shape().elem_count()
    }

    fn uniforms(&self) -> KernelUniforms {
        KernelUniforms::new(
            self.elem_count(),
            self.out_layout,
            self.in0_layout,
            self.in1_layout,
        )
    }
}

impl WgpuKernel {
    pub fn compile(device: &WgpuDevice, source: &str, entry_point: &str) -> Result<Self> {
        Self::compile_with_workgroup_size(
            device,
            source,
            entry_point,
            device.caps().elem_workgroup_size,
        )
    }

    pub fn compile_with_workgroup_size(
        device: &WgpuDevice,
        source: &str,
        entry_point: &str,
        workgroup_size: u32,
    ) -> Result<Self> {
        let bind_group_builder = BindGroupBuilder::new();
        let layout = bind_group_builder
            .bind_group_layout()
            .get_or_create(device.device())?;
        let pipeline = device.shader_cache().get_or_create_pipeline_with_layout(
            device.device(),
            Some(&layout),
            STANDARD_KERNEL_LAYOUT_KEY,
            source,
            entry_point,
        )?;

        Ok(Self {
            pipeline,
            bind_group_builder,
            extended_bind_group_builder: None,
            sdpa_bind_group_builder: None,
            workgroup_size,
        })
    }

    pub fn compile_extended(
        device: &WgpuDevice,
        source: &str,
        entry_point: &str,
        workgroup_size: u32,
    ) -> Result<Self> {
        let extended_bind_group_builder = ExtendedBindGroupBuilder::new();
        let layout = extended_bind_group_builder
            .bind_group_layout()
            .get_or_create(device.device())?;
        let pipeline = device.shader_cache().get_or_create_pipeline_with_layout(
            device.device(),
            Some(&layout),
            EXTENDED_KERNEL_LAYOUT_KEY,
            source,
            entry_point,
        )?;

        Ok(Self {
            pipeline,
            bind_group_builder: BindGroupBuilder::new(),
            extended_bind_group_builder: Some(extended_bind_group_builder),
            sdpa_bind_group_builder: None,
            workgroup_size,
        })
    }

    pub fn compile_sdpa(
        device: &WgpuDevice,
        source: &str,
        entry_point: &str,
        workgroup_size: u32,
    ) -> Result<Self> {
        let sdpa_bind_group_builder = SdpaBindGroupBuilder::new();
        let layout = sdpa_bind_group_builder
            .bind_group_layout()
            .get_or_create(device.device())?;
        let pipeline = device.shader_cache().get_or_create_pipeline_with_layout(
            device.device(),
            Some(&layout),
            SDPA_KERNEL_LAYOUT_KEY,
            source,
            entry_point,
        )?;

        Ok(Self {
            pipeline,
            bind_group_builder: BindGroupBuilder::new(),
            extended_bind_group_builder: None,
            sdpa_bind_group_builder: Some(sdpa_bind_group_builder),
            workgroup_size,
        })
    }

    /// Builds a kernel handle for random-number shaders (custom bind group layout).
    pub fn from_random_pipeline(
        pipeline: wgpu::ComputePipeline,
        workgroup_size: u32,
    ) -> Self {
        Self {
            pipeline,
            bind_group_builder: BindGroupBuilder::new(),
            extended_bind_group_builder: None,
            sdpa_bind_group_builder: None,
            workgroup_size,
        }
    }

    pub fn compile_with_cache(
        wgpu_device: &wgpu::Device,
        shader_cache: &ShaderCache,
        bind_group_layout: &StandardBindGroupLayout,
        source: &str,
        entry_point: &str,
        workgroup_size: u32,
    ) -> Result<Self> {
        let layout = bind_group_layout.get_or_create(wgpu_device)?;
        let pipeline = shader_cache.get_or_create_pipeline_with_layout(
            wgpu_device,
            Some(&layout),
            STANDARD_KERNEL_LAYOUT_KEY,
            source,
            entry_point,
        )?;
        Ok(Self {
            pipeline,
            bind_group_builder: BindGroupBuilder::new(),
            extended_bind_group_builder: None,
            sdpa_bind_group_builder: None,
            workgroup_size,
        })
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    pub fn workgroup_count(&self, elem_count: usize) -> u32 {
        workgroup_count(self.workgroup_size, elem_count)
    }

    pub fn encode_dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        args: KernelDispatchArgs<'_>,
    ) -> Result<()> {
        let elem_count = args.elem_count();
        let uniforms = args.uniforms();
        let bind_group = self.bind_group_builder.create_bind_group(
            device,
            queue,
            StandardBindGroupArgs {
                output: args.output,
                input0: args.input0,
                input1: args.input1,
                uniforms: &uniforms,
            },
        )?;

        let workgroups = self.workgroup_count(elem_count);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("candle wgpu kernel"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        Ok(())
    }

    pub fn dispatch(&self, device: &WgpuDevice, args: KernelDispatchArgs<'_>) -> Result<()> {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("candle wgpu kernel dispatch"),
            });
        self.encode_dispatch(device.device(), device.queue(), &mut encoder, args)?;
        device.queue().submit(Some(encoder.finish()));
        poll_device(device.device())?;
        Ok(())
    }

    pub fn encode_dispatch_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("candle wgpu kernel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    pub fn create_extended_bind_group(
        &self,
        device: &WgpuDevice,
        output: BufferOffset<'_>,
        input0: BufferOffset<'_>,
        input1: BufferOffset<'_>,
        input2: BufferOffset<'_>,
        uniform_bytes: &[u8],
    ) -> Result<wgpu::BindGroup> {
        let builder = self
            .extended_bind_group_builder
            .as_ref()
            .ok_or_else(|| super::error::WgpuError::Message("kernel is not extended".into()))?;
        builder.create_bind_group(
            device.device(),
            device.queue(),
            ExtendedBindGroupArgs {
                output,
                input0,
                input1,
                input2,
                uniform_bytes,
            },
        )
    }

    pub fn create_sdpa_bind_group(
        &self,
        device: &WgpuDevice,
        output: BufferOffset<'_>,
        q: BufferOffset<'_>,
        k: BufferOffset<'_>,
        v: BufferOffset<'_>,
        mask: BufferOffset<'_>,
        uniform_bytes: &[u8],
    ) -> Result<wgpu::BindGroup> {
        let builder = self
            .sdpa_bind_group_builder
            .as_ref()
            .ok_or_else(|| super::error::WgpuError::Message("kernel is not sdpa".into()))?;
        builder.create_bind_group(
            device.device(),
            device.queue(),
            SdpaBindGroupArgs {
                output,
                q,
                k,
                v,
                mask,
                uniform_bytes,
            },
        )
    }

    pub fn dispatch_bind_group(
        &self,
        device: &WgpuDevice,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) -> Result<()> {
        let mut encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("candle wgpu kernel dispatch"),
            });
        self.encode_dispatch_bind_group(&mut encoder, bind_group, workgroups);
        device.queue().submit(Some(encoder.finish()));
        poll_device(device.device())?;
        Ok(())
    }
}

impl WgpuDevice {
    pub fn compile_kernel(&self, source: &str, entry_point: &str) -> Result<WgpuKernel> {
        self.compile_tuned_kernel(source, entry_point)
    }

    /// Compile a kernel with Intel-tuned workgroup sizes injected into WGSL source.
    pub fn compile_tuned_kernel(&self, source: &str, entry_point: &str) -> Result<WgpuKernel> {
        let tuned = tune_shader_source(source, self.caps());
        WgpuKernel::compile_with_workgroup_size(
            self,
            &tuned,
            entry_point,
            self.caps().elem_workgroup_size,
        )
    }
}

fn workgroup_count(workgroup_size: u32, elem_count: usize) -> u32 {
    let size = workgroup_size.max(1);
    (elem_count as u32).div_ceil(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_count_rounds_up() {
        assert_eq!(workgroup_count(8, 0), 0);
        assert_eq!(workgroup_count(8, 1), 1);
        assert_eq!(workgroup_count(8, 8), 1);
        assert_eq!(workgroup_count(8, 9), 2);
    }
}
