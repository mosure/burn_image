use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::{num::NonZeroU64, sync::Arc};
use wgpu::BufferUsages;

#[cfg(target_family = "wasm")]
#[path = "storage_release.rs"]
mod storage_release;

#[cfg(target_family = "wasm")]
fn release_deallocated_buffer(buffer: wgpu::Buffer, resource_guard: Arc<()>) {
    // Dropping wgpu's WebGPU buffer wrapper only releases its JavaScript reference and leaves
    // actual GPUBuffer reclamation to garbage collection. A free CubeCL managed handle does not,
    // however, prove that no queued cross-stream task still owns a raw `WgpuResource` clone. Only
    // destroy at an alias-free storage boundary; otherwise preserve the wrapper's GC semantics so
    // queued or encoded work cannot be invalidated before submission.
    storage_release::release_resource_before_drop_if_unaliased(
        buffer,
        resource_guard,
        wgpu::Buffer::destroy,
    );
}

#[cfg(not(target_family = "wasm"))]
fn release_deallocated_buffer(buffer: wgpu::Buffer, resource_guard: Arc<()>) {
    // Preserve the upstream native backend's ordinary RAII release behavior.
    drop(resource_guard);
    drop(buffer);
}

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, (wgpu::Buffer, Arc<()>)>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
    /// Keeps the storage allocation from being explicitly destroyed while a raw resource alias is
    /// queued, encoded, or exposed through a managed resource.
    _resource_guard: Arc<()>,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(mem_alignment: usize, device: wgpu::Device, usages: BufferUsages) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let (buffer, resource_guard) = self.memory.get(&handle.id).unwrap();
        WgpuResource::new(
            buffer.clone(),
            handle.offset(),
            handle.size(),
            resource_guard.clone(),
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        });

        self.memory.insert(id, (buffer, Arc::new(())));
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        if let Some((buffer, resource_guard)) = self.memory.remove(&id) {
            release_deallocated_buffer(buffer, resource_guard);
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }
}
