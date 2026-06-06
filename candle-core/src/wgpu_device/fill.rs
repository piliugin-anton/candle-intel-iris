use super::bind_group::{BindGroupBuilder, KernelUniforms};
use super::error::Result;
use super::kernel::WgpuKernel;
use super::storage::{BufferOffset, WgpuStorage, STORAGE_BUFFER_USAGE};
use crate::backend::BackendStorage;
use crate::scalar::Scalar;
use crate::wgsl::{CONST_SET_BF16, CONST_SET_F16, CONST_SET_F32, CONST_SET_U32, CONST_SET_U8};
use crate::{DType, Error, Layout, Result as CandleResult};

fn compile_const_set_kernel(
    device: &super::WgpuDevice,
    source: &str,
    entry: &str,
) -> Result<WgpuKernel> {
    let tuned = super::intel_caps::tune_shader_source(source, device.caps());
    WgpuKernel::compile_with_workgroup_size(device, &tuned, entry, device.caps().elem_workgroup_size)
}

fn dispatch_const_set_kernel(
    storage: &mut WgpuStorage,
    layout: &Layout,
    source: &str,
    entry: &str,
    value_bits: u32,
) -> Result<()> {
    let device = storage.device();
    let elem_count = layout.shape().elem_count();
    let uniforms = KernelUniforms::new_const_set(elem_count, layout, value_bits);
    let kernel = compile_const_set_kernel(device, source, entry)?;
    // In-place fill only writes binding 0; dummy read-only slots avoid buffer usage conflicts.
    let dummy_buf = device.allocate_buffer(16, STORAGE_BUFFER_USAGE)?;
    let dummy = BufferOffset {
        buffer: dummy_buf.as_ref(),
        offset_in_bytes: 0,
    };
    // Layout start offset lives in the uniform block; bind at 0 for wgpu alignment rules.
    let output = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: 0,
    };
    let bind_group = BindGroupBuilder::new().create_bind_group_bytes(
        device.device(),
        device.queue(),
        output,
        dummy.clone(),
        Some(dummy),
        uniforms.as_bytes(),
    )?;
    let wg = device.caps().elem_workgroup_size;
    let grid = (elem_count as u32).div_ceil(wg);
    storage
        .backing()
        .with_unmapped(|| kernel.dispatch_bind_group(device, &bind_group, [grid, 1, 1]))?;
    Ok(())
}

pub fn const_set_via_cpu(
    storage: &mut WgpuStorage,
    layout: &Layout,
    scalar: Scalar,
) -> CandleResult<()> {
    let mut cpu = storage.to_cpu_storage()?;
    cpu.const_set(scalar, layout)?;
    *storage = WgpuStorage::from_cpu(storage.device(), &cpu).map_err(Error::from)?;
    Ok(())
}

pub fn dispatch_const_set(
    storage: &mut WgpuStorage,
    layout: &Layout,
    scalar: Scalar,
) -> CandleResult<()> {
    if scalar.dtype() != storage.dtype() {
        return Err(Error::Msg(format!(
            "const_set dtype mismatch, expected {:?} but got {:?}",
            storage.dtype(),
            scalar.dtype()
        ))
        .bt());
    }

    let device = storage.device();
    match (storage.dtype(), scalar) {
        (DType::F32, Scalar::F32(v)) => {
            dispatch_const_set_kernel(
                storage,
                layout,
                CONST_SET_F32,
                "const_set_f32",
                v.to_bits(),
            )
            .map_err(Error::from)
        }
        (DType::U32, Scalar::U32(v)) => dispatch_const_set_kernel(
            storage,
            layout,
            CONST_SET_U32,
            "const_set_u32",
            v,
        )
        .map_err(Error::from),
        (DType::U8, Scalar::U8(v)) => dispatch_const_set_kernel(
            storage,
            layout,
            CONST_SET_U8,
            "const_set_u8",
            v as u32,
        )
        .map_err(Error::from),
        (DType::F16, Scalar::F16(v)) if device.caps().supports_native_f16() => {
            dispatch_const_set_kernel(
                storage,
                layout,
                CONST_SET_F16,
                "const_set_f16",
                f32::from(v).to_bits(),
            )
            .map_err(Error::from)
        }
        (DType::BF16, Scalar::BF16(v)) if device.caps().supports_native_bf16() => {
            dispatch_const_set_kernel(
                storage,
                layout,
                CONST_SET_BF16,
                "const_set_bf16",
                u32::from(v.to_bits()),
            )
            .map_err(Error::from)
        }
        (DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0, _) => Err(
            Error::Msg(format!(
                "const_set not supported for dummy type {:?}",
                storage.dtype()
            ))
            .bt(),
        ),
        _ => const_set_via_cpu(storage, layout, scalar),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_device::WgpuDevice;

    #[test]
    fn const_set_shaders_define_entry_points() {
        assert!(CONST_SET_F32.contains("fn const_set_f32"));
        assert!(CONST_SET_U32.contains("fn const_set_u32"));
        assert!(CONST_SET_U8.contains("fn const_set_u8"));
    }

    #[test]
    fn const_set_f32_kernel_compiles_on_noop_device() {
        let device = WgpuDevice::new_test(false, 4096);
        compile_const_set_kernel(&device, CONST_SET_F32, "const_set_f32")
            .expect("const_set_f32 compiles");
    }
}
