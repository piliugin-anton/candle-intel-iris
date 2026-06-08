use super::async_io::wait_for_buffer_map;
use super::error::{Result, WgpuError};
use super::WgpuDevice;
use std::sync::{Arc, Mutex};
use wgpu::BufferUsages;

/// Staging buffer usage for CPU readback (WebGPU: `MAP_READ` pairs with `COPY_DST` only).
pub const MAPPED_READ_USAGE: BufferUsages = BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

/// Tracks whether a mappable staging buffer is currently CPU-mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapState {
    Unmapped,
    MappedWrite,
    MappedRead,
}

/// A wgpu buffer that supports CPU mapping for readback.
#[derive(Debug, Clone)]
struct MappedStaging {
    buffer: Arc<wgpu::Buffer>,
    map_state: Arc<Mutex<MapState>>,
}

impl MappedStaging {
    fn new(buffer: Arc<wgpu::Buffer>, initially_mapped: bool) -> Self {
        Self {
            buffer,
            map_state: Arc::new(Mutex::new(if initially_mapped {
                MapState::MappedWrite
            } else {
                MapState::Unmapped
            })),
        }
    }

    fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    fn unmap(&self) -> Result<()> {
        let mut state = self
            .map_state
            .lock()
            .map_err(|e| WgpuError::LockPoisoned(e.to_string()))?;
        if *state != MapState::Unmapped {
            self.buffer.unmap();
            *state = MapState::Unmapped;
        }
        Ok(())
    }

    fn map_async_read(&self, device: &wgpu::Device) -> Result<()> {
        self.unmap()?;
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        wait_for_buffer_map(device, &rx)?;
        let mut state = self
            .map_state
            .lock()
            .map_err(|e| WgpuError::LockPoisoned(e.to_string()))?;
        *state = MapState::MappedRead;
        Ok(())
    }

    fn read_bytes_region(&self, device: &wgpu::Device, size: u64) -> Result<Vec<u8>> {
        self.map_async_read(device)?;
        let slice = self.buffer.slice(..size);
        let mapped = slice.get_mapped_range();
        let bytes = mapped.to_vec();
        drop(mapped);
        self.unmap()?;
        Ok(bytes)
    }

    fn with_unmapped<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.unmap()?;
        f()
    }
}

/// UMA-friendly tensor backing: GPU storage buffer plus a pooled read-staging buffer.
///
/// WebGPU forbids combining `STORAGE` with `MAP_*`, so zero-copy readback reuses a
/// dedicated `MAP_READ | COPY_DST` staging buffer instead of mapping the storage buffer.
#[derive(Debug, Clone)]
pub struct MappedBacking {
    storage: Arc<wgpu::Buffer>,
    read_staging: MappedStaging,
}

fn aligned_copy_size(size: usize) -> usize {
    let align = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    size.div_ceil(align) * align
}

impl MappedBacking {
    pub fn new(device: &WgpuDevice, byte_len: usize) -> Result<Self> {
        let storage_usage = BufferUsages::STORAGE
            .union(BufferUsages::COPY_DST)
            .union(BufferUsages::COPY_SRC);
        let storage = device.allocate_buffer(byte_len, storage_usage)?;
        let staging = device.allocator().allocate_mapped(
            device.device(),
            aligned_copy_size(byte_len),
            MAPPED_READ_USAGE,
            false,
        )?;
        Ok(Self {
            storage,
            read_staging: MappedStaging::new(staging, false),
        })
    }

    pub fn storage(&self) -> &Arc<wgpu::Buffer> {
        &self.storage
    }

    pub fn read_bytes_region(
        &self,
        device: &WgpuDevice,
        src_byte_offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>> {
        let wgpu_device = device.device();
        let queue = device.queue();
        let (copy_offset, copy_size, head_skip, out_len) =
            super::async_io::copy_aligned_range(src_byte_offset as usize, byte_len as usize);
        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu mapped readback"),
        });
        encoder.copy_buffer_to_buffer(
            &self.storage,
            copy_offset,
            self.read_staging.buffer(),
            0,
            copy_size,
        );
        queue.submit(Some(encoder.finish()));
        let mut bytes = self
            .read_staging
            .read_bytes_region(wgpu_device, copy_size)?;
        bytes.drain(..head_skip);
        bytes.truncate(out_len);
        Ok(bytes)
    }

    pub fn with_unmapped<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.read_staging.with_unmapped(f)
    }
}
