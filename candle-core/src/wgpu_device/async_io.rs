use super::error::{Result, WgpuError};
use std::sync::mpsc::Receiver;

/// Poll the device until a buffer `map_async` callback delivers its result.
pub fn wait_for_buffer_map(
    device: &wgpu::Device,
    rx: &Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
) -> Result<()> {
    loop {
        poll_device(device)?;
        if let Ok(result) = rx.try_recv() {
            return result.map_err(WgpuError::from);
        }
    }
}

/// Non-blocking poll after queue submit — lets the driver batch work across ops.
pub fn poll_device_progress(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::Poll)
        .map_err(WgpuError::from)?;
    Ok(())
}

/// Block until all in-flight queue work completes.
pub fn poll_device(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(WgpuError::from)?;
    Ok(())
}
