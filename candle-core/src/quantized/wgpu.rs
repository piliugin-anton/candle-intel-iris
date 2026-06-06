use super::{GgmlDType, QStorage};
use crate::custom_op::CustomOp1;
use crate::quantized::k_quants::GgmlType;
use crate::wgpu_device::{
    dispatch_dequant_f32, dispatch_qmatmul_q4_0, dispatch_qmatmul_q4_k, dispatch_qmatmul_q5_0,
    dispatch_qmatmul_q8_0, dispatch_quant_f32, gpu_dequant_supported, gpu_quant_supported,
    upload_quant_weights, wait_for_buffer_map, WgpuStorage, STORAGE_BUFFER_USAGE,
};
use crate::{backend::BackendStorage, CpuStorage, DType, Layout, Result, Shape, WgpuDevice};
use std::sync::Arc;
use wgpu::BufferUsages;

#[derive(Clone, Debug)]
pub struct QWgpuStorage {
    dtype: GgmlDType,
    device: WgpuDevice,
    buffer: Arc<wgpu::Buffer>,
    size_in_bytes: usize,
}

impl QWgpuStorage {
    pub fn zeros(device: &WgpuDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = elem_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle qtensor zeros"),
            size: size_in_bytes as u64,
            usage: STORAGE_BUFFER_USAGE,
            mapped_at_creation: false,
        });
        Ok(Self {
            dtype,
            device: device.clone(),
            buffer: Arc::new(buffer),
            size_in_bytes,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<WgpuStorage> {
        if gpu_dequant_supported(self.dtype) {
            return dispatch_dequant_f32(
                &self.device,
                self.dtype,
                &self.buffer,
                elem_count,
            )
            .map_err(Into::into);
        }
        let mut out = vec![0f32; elem_count];
        dequantize_to_f32(self, &mut out)?;
        WgpuStorage::from_cpu(&self.device, &CpuStorage::F32(out)).map_err(Into::into)
    }

    pub fn quantize(&mut self, src: &WgpuStorage) -> Result<()> {
        if gpu_quant_supported(self.dtype) {
            let elem_count = src.elem_count();
            let layout = Layout::contiguous(&Shape::from(elem_count));
            let buffer = dispatch_quant_f32(&self.device, self.dtype, src, &layout)?;
            self.size_in_bytes = elem_count / self.dtype.block_size() * self.dtype.type_size();
            self.buffer = buffer;
            return Ok(());
        }
        let src = src.to_cpu_storage()?;
        let elem_count = match &src {
            CpuStorage::F32(v) => v.len(),
            _ => crate::bail!("wgpu quantize expects f32 activations"),
        };
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        self.upload_bytes(&qcpu_storage.data()?)?;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &WgpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src = src.to_cpu_storage()?;
        let elem_count = match &src {
            CpuStorage::F32(v) => v.len(),
            _ => crate::bail!("wgpu quantize expects f32 activations"),
        };
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        self.upload_bytes(&qcpu_storage.data()?)?;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }
        self.upload_bytes(&qcpu_storage.data()?)?;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }
        self.upload_bytes(&qcpu_storage.data()?)?;
        Ok(())
    }

    fn upload_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.size_in_bytes {
            let buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle qtensor"),
                size: bytes.len() as u64,
                usage: STORAGE_BUFFER_USAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.buffer = Arc::new(buffer);
            self.size_in_bytes = bytes.len();
        }
        self.device.queue().write_buffer(&self.buffer, 0, bytes);
        Ok(())
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        match self.dtype {
            GgmlDType::Q4_0 | GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::Q4K => {
                self.fwd_qmatmul(self_shape, storage, layout)
            }
            _ => self.fwd_via_cpu(self_shape, storage, layout),
        }
    }

    fn fwd_qmatmul(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        if storage.dtype() != DType::F32 {
            crate::bail!("wgpu qmatmul only supports f32 activations, got {:?}", storage.dtype());
        }

        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }

        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            rank => crate::bail!("wgpu qmatmul unsupported input rank {rank}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        let out = match self.dtype {
            GgmlDType::Q4_0 => dispatch_qmatmul_q4_0(storage, &self.buffer, (1, m, n, k), layout)?,
            GgmlDType::Q5_0 => dispatch_qmatmul_q5_0(storage, &self.buffer, (1, m, n, k), layout)?,
            GgmlDType::Q8_0 => dispatch_qmatmul_q8_0(storage, &self.buffer, (1, m, n, k), layout)?,
            GgmlDType::Q4K => dispatch_qmatmul_q4_k(storage, &self.buffer, (1, m, n, k), layout)?,
            other => crate::bail!("unsupported wgpu qmatmul dtype {other:?}"),
        };
        Ok((out, dst_shape))
    }

    fn fwd_via_cpu(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        let bytes = self.data()?;
        let qtensor = super::QTensor::new(
            QStorage::from_data(std::borrow::Cow::from(bytes), &crate::Device::Cpu, self.dtype)?,
            self_shape,
        )?;
        let (cpu_out, dst_shape) = qtensor.cpu_fwd(&storage.to_cpu_storage()?, layout)?;
        let out = match cpu_out {
            CpuStorage::F32(v) => WgpuStorage::from_cpu(&self.device, &CpuStorage::F32(v))?,
            CpuStorage::F16(v) => WgpuStorage::from_cpu(&self.device, &CpuStorage::F16(v))?,
            other => crate::bail!("unexpected qmatmul output dtype {:?}", other),
        };
        Ok((out, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        read_buffer_bytes(&self.device, &self.buffer, self.size_in_bytes)
    }
}

pub fn load_quantized<T: GgmlType + Send + Sync + 'static>(
    device: &WgpuDevice,
    data: &[T],
) -> Result<QStorage> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data))
    };
    let buffer = upload_quant_weights(device, bytes)?;
    Ok(QStorage::Wgpu(Box::new(QWgpuStorage {
        dtype: T::DTYPE,
        device: device.clone(),
        buffer,
        size_in_bytes: bytes.len(),
    })))
}

fn dequantize_to_f32(storage: &QWgpuStorage, out: &mut [f32]) -> Result<()> {
    let bytes = storage.data()?;
    match storage.dtype {
        GgmlDType::F32 => {
            let slice = as_t_slice::<f32>(&bytes);
            f32::to_float(slice, out);
        }
        GgmlDType::F16 => {
            let slice = as_t_slice::<half::f16>(&bytes);
            half::f16::to_float(slice, out);
        }
        GgmlDType::BF16 => {
            let slice = as_t_slice::<half::bf16>(&bytes);
            half::bf16::to_float(slice, out);
        }
        GgmlDType::Q4_0 => {
            let slice = as_t_slice::<crate::quantized::BlockQ4_0>(&bytes);
            crate::quantized::BlockQ4_0::to_float(slice, out);
        }
        GgmlDType::Q4_1 => {
            let slice = as_t_slice::<crate::quantized::BlockQ4_1>(&bytes);
            crate::quantized::BlockQ4_1::to_float(slice, out);
        }
        GgmlDType::Q5_0 => {
            let slice = as_t_slice::<crate::quantized::BlockQ5_0>(&bytes);
            crate::quantized::BlockQ5_0::to_float(slice, out);
        }
        GgmlDType::Q5_1 => {
            let slice = as_t_slice::<crate::quantized::BlockQ5_1>(&bytes);
            crate::quantized::BlockQ5_1::to_float(slice, out);
        }
        GgmlDType::Q8_0 => {
            let slice = as_t_slice::<crate::quantized::BlockQ8_0>(&bytes);
            crate::quantized::BlockQ8_0::to_float(slice, out);
        }
        GgmlDType::Q8_1 => {
            let slice = as_t_slice::<crate::quantized::BlockQ8_1>(&bytes);
            crate::quantized::BlockQ8_1::to_float(slice, out);
        }
        GgmlDType::Q2K => {
            let slice = as_t_slice::<crate::quantized::BlockQ2K>(&bytes);
            crate::quantized::BlockQ2K::to_float(slice, out);
        }
        GgmlDType::Q3K => {
            let slice = as_t_slice::<crate::quantized::BlockQ3K>(&bytes);
            crate::quantized::BlockQ3K::to_float(slice, out);
        }
        GgmlDType::Q4K => {
            let slice = as_t_slice::<crate::quantized::BlockQ4K>(&bytes);
            crate::quantized::BlockQ4K::to_float(slice, out);
        }
        GgmlDType::Q5K => {
            let slice = as_t_slice::<crate::quantized::BlockQ5K>(&bytes);
            crate::quantized::BlockQ5K::to_float(slice, out);
        }
        GgmlDType::Q6K => {
            let slice = as_t_slice::<crate::quantized::BlockQ6K>(&bytes);
            crate::quantized::BlockQ6K::to_float(slice, out);
        }
        GgmlDType::Q8K => {
            let slice = as_t_slice::<crate::quantized::BlockQ8K>(&bytes);
            crate::quantized::BlockQ8K::to_float(slice, out);
        }
    }
    Ok(())
}

fn as_t_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(data.len() % size, 0);
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len() / size) }
}

fn read_buffer_bytes(device: &WgpuDevice, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<u8>> {
    let wgpu_device = device.device();
    let queue = device.queue();
    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("candle qtensor readback"),
        size: size as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("candle qtensor readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    wait_for_buffer_map(wgpu_device, &rx)?;
    let mapped = slice.get_mapped_range();
    Ok(mapped.to_vec())
}
