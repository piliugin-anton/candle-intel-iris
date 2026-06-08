use super::error::{Result, WgpuError};
use std::sync::mpsc::Receiver;

/// Expand `(byte_offset, byte_len)` to wgpu `copy_buffer_to_buffer` alignment requirements.
///
/// Returns `(copy_offset, copy_size, head_skip, output_len)`.
pub(crate) fn copy_aligned_range(
    byte_offset: usize,
    byte_len: usize,
) -> (u64, u64, usize, usize) {
    const ALIGN: usize = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    let aligned_off = byte_offset / ALIGN * ALIGN;
    let head_skip = byte_offset - aligned_off;
    let total = head_skip + byte_len;
    let aligned_len = total.div_ceil(ALIGN) * ALIGN;
    (
        aligned_off as u64,
        aligned_len as u64,
        head_skip,
        byte_len,
    )
}

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

#[cfg(test)]
mod tests {
    use super::copy_aligned_range;

    #[test]
    fn copy_aligned_range_expands_unaligned_f16_slice() {
        let (off, size, head, out) = copy_aligned_range(2, 4);
        assert_eq!(off, 0);
        assert_eq!(size, 8);
        assert_eq!(head, 2);
        assert_eq!(out, 4);
    }

    #[test]
    fn copy_aligned_range_keeps_aligned_f32_slice() {
        let (off, size, head, out) = copy_aligned_range(8, 12);
        assert_eq!(off, 8);
        assert_eq!(size, 12);
        assert_eq!(head, 0);
        assert_eq!(out, 12);
    }
}
