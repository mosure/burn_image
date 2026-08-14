# CubeCL WGPU Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) WGPU runtime.

## Workspace patch

This repository carries browser/native correctness fixes on top of the published `0.10.0`
crate. An upload-only stream now submits its pending `wgpu::Queue::write_buffer` work when CubeCL
is asked to synchronize, even if no compute task has been queued. The upstream early return
considered only compute tasks, so model-loading staging buffers could remain pending across many
authenticated shards. The patched synchronization is a real upload-completion boundary and lets
the native low-VRAM loader safely release transient allocator pages per physical shard.

On Wasm, allocator deallocation now explicitly destroys an alias-free removed `GPUBuffer` before
dropping its Rust wrapper. The upstream WebGPU wrapper intentionally has a no-op `Drop` because
buffers are cloneable, but that otherwise leaves large, fully unreferenced allocations to
JavaScript garbage collection. CubeCL can drop a managed binding while a cross-stream task still
owns a raw resource clone, so aliased buffers retain the existing GC behavior instead of being
invalidated before submission. Synchronized, alias-free request phase boundaries get predictable
reclamation while native backends retain their existing RAII behavior.

Every new shader module, bind-group/pipeline layout, compute pipeline, WGPU task/write operation,
and actual queue submission is wrapped in validation, out-of-memory, and internal error scopes.
Browser pipeline-creation futures retain the exact entrypoint on the launch stream; all detached
futures survive CubeCL's ordinary bounded 32-task auto-submissions and are drained by the next
explicit synchronization. Native pipeline-creation errors remain synchronous. This covers errors
raised during pipeline creation, while bindings and commands are encoded, and asynchronously by
submitted work without accumulating an entire model stage in one command buffer. A source-level
unit test locks the empty, upload-only, compute-only, and mixed submission cases.

Runtime execution limits and optional features are registered from the logical `wgpu::Device`, not
the physical adapter. This matters for a device shared with a Bevy frontend: the requested device
may deliberately expose lower workgroup/binding limits and fewer features than its adapter can
support. CubeCL now validates launches against those requested limits and registers timestamp,
subgroup, WGSL scalar/atomic, and mapped Metal features only when the device enabled them. A Wasm
device without `TIMESTAMP_QUERY` remains usable for ordinary inference; attempts to profile it fail
clearly instead of panicking at stream construction or blocking the browser event loop. The native
SPIR-V path retains CubeCL 0.10.0's raw Vulkan extension registration for compatibility; callers
supplying an external native Vulkan device must still ensure its raw enabled-feature chain matches
the adapter capabilities CubeCL registers. That residual native passthrough caveat does not apply
to the browser WGSL path.

## Configuration

You can set `CUBECL_WGPU_MAX_TASKS` to a positive integer that determines how many computing tasks are submitted in batches to the graphics API.

## Platform Support

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |
