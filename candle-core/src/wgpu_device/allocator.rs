use super::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    bucket_size: usize,
    usage: u32,
}

type BufferMap = HashMap<PoolKey, Vec<Arc<wgpu::Buffer>>>;

/// Size-bucketed buffer pool to reduce allocation overhead.
///
/// Buffers are grouped by allocated size (next power of two) and usage flags so
/// mapped and device-local buffers are never reused interchangeably.
#[derive(Clone)]
pub struct Allocator {
    buffers: Arc<RwLock<BufferMap>>,
}

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocator").finish_non_exhaustive()
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn allocate(
        &self,
        device: &wgpu::Device,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> Result<Arc<wgpu::Buffer>> {
        self.allocate_mapped(device, size, usage, false)
    }

    pub fn allocate_mapped(
        &self,
        device: &wgpu::Device,
        size: usize,
        usage: wgpu::BufferUsages,
        mapped_at_creation: bool,
    ) -> Result<Arc<wgpu::Buffer>> {
        let key = pool_key(size, usage);
        {
            let buffers = self.buffers.write()?;
            if let Some(buffer) = find_available_buffer(key, &buffers) {
                return Ok(buffer);
            }
        }

        let bucket_size = key.bucket_size;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bucket_size as u64,
            usage,
            mapped_at_creation,
        });
        let buffer = Arc::new(buffer);
        let mut buffers = self.buffers.write()?;
        buffers.entry(key).or_default().push(buffer.clone());
        Ok(buffer)
    }

    pub fn drop_unused(&self) -> Result<()> {
        let mut buffers = self.buffers.write()?;
        for subbuffers in buffers.values_mut() {
            subbuffers.retain(|buffer| Arc::strong_count(buffer) > 1);
        }
        buffers.retain(|_, subbuffers| !subbuffers.is_empty());
        Ok(())
    }
}

fn pool_key(size: usize, usage: wgpu::BufferUsages) -> PoolKey {
    PoolKey {
        bucket_size: bucket_size(size),
        usage: usage.bits(),
    }
}

fn bucket_size(size: usize) -> usize {
    size.next_power_of_two().max(1)
}

fn find_available_buffer(key: PoolKey, buffers: &BufferMap) -> Option<Arc<wgpu::Buffer>> {
    let mut best_buffer: Option<&Arc<wgpu::Buffer>> = None;
    let mut best_bucket = usize::MAX;
    for (pool_key, subbuffers) in buffers.iter() {
        if pool_key.usage != key.usage {
            continue;
        }
        if pool_key.bucket_size >= key.bucket_size && pool_key.bucket_size < best_bucket {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_bucket = pool_key.bucket_size;
                }
            }
        }
    }
    best_buffer.cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::BufferUsages;

    #[test]
    fn test_bucket_size_exact_powers_of_two() {
        assert_eq!(bucket_size(1), 1);
        assert_eq!(bucket_size(2), 2);
        assert_eq!(bucket_size(4), 4);
        assert_eq!(bucket_size(1024), 1024);
    }

    #[test]
    fn test_bucket_size_rounds_up() {
        assert_eq!(bucket_size(3), 4);
        assert_eq!(bucket_size(5), 8);
        assert_eq!(bucket_size(1000), 1024);
    }

    #[test]
    fn pool_key_distinguishes_usage() {
        let a = pool_key(64, BufferUsages::STORAGE);
        let b = pool_key(64, BufferUsages::STORAGE | BufferUsages::MAP_READ);
        assert_ne!(a, b);
    }
}
