#[cfg(target_family = "wasm")]
use super::stream::PipelineCacheInvalidation;
use super::{
    storage::{WgpuResource, WgpuStorage},
    stream::WgpuErrorScopes,
};
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use crate::{AutoCompiler, AutoRepresentation};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use cubecl_core::server::{Binding, StreamErrorMode};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
#[cfg(feature = "spirv")]
use cubecl_core::{cache::CacheOption, compilation_cache::CompilationCache, hash::StableHash};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::memory_management::{ManagedMemoryHandle, MemoryUsage};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    validation::{validate_cube_dim, validate_units},
};
use hashbrown::HashMap;
use wgpu::{ComputePipeline, ErrorFilter};

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    /// Set when an asynchronous WebGPU module/pipeline creation scope reports an error.
    /// The next Wasm launch consumes the marker and drops every potentially invalid pipeline.
    #[cfg(target_family = "wasm")]
    pipeline_cache_invalidation: PipelineCacheInvalidation,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache:
        Option<CompilationCache<(u64, StableHash), cubecl_spirv::SpirvCacheEntry>>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines: HashMap::new(),
            #[cfg(target_family = "wasm")]
            pipeline_cache_invalidation: PipelineCacheInvalidation::default(),
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache: {
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "spirv",
                        CacheOption::default().name("vulkan").root(root),
                    ))
                } else {
                    None
                }
            },
            backend,
            utilities: Arc::new(utilities),
        }
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.into_iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b)?;
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
        })
    }

    fn begin_pipeline_error_scopes(&mut self, stream_id: &StreamId) -> WgpuErrorScopes {
        self.scheduler
            .stream(stream_id)
            .begin_pipeline_error_scopes()
    }

    fn finish_pipeline_error_scopes(
        &mut self,
        stream_id: &StreamId,
        scopes: WgpuErrorScopes,
        context: Arc<str>,
    ) -> Result<(), LaunchError> {
        #[cfg(target_family = "wasm")]
        let pipeline_cache_invalidation = Some(self.pipeline_cache_invalidation.clone());
        #[cfg(not(target_family = "wasm"))]
        let pipeline_cache_invalidation = None;

        let errors = self
            .scheduler
            .stream(stream_id)
            .finish_pipeline_error_scopes(scopes, context.clone(), pipeline_cache_invalidation);

        pipeline_creation_error(errors, &context).map_or(Ok(()), Err)
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
        stream_id: &StreamId,
    ) -> Result<Arc<ComputePipeline>, LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        #[cfg(target_family = "wasm")]
        if self.pipeline_cache_invalidation.take() {
            log::warn!(
                "Discarding cached WGPU pipelines after an asynchronous WebGPU pipeline-creation error"
            );
            self.pipelines.clear();
        }

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        #[cfg(feature = "spirv")]
        let cached = {
            let scopes = self.begin_pipeline_error_scopes(stream_id);
            let result = self.load_cached_pipeline(&kernel_id, bindings, mode);
            let context = match &result {
                Ok(Some(Ok((_, entrypoint)))) => Arc::from(format!(
                    "while creating cached shader module/layout/compute pipeline for entrypoint `{entrypoint}`"
                )),
                _ => Arc::from(format!(
                    "while looking up or creating cached compute pipeline for kernel `{kernel_id:?}`"
                )),
            };
            self.finish_pipeline_error_scopes(stream_id, scopes, context)?;
            result?
        };

        #[cfg(not(feature = "spirv"))]
        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok((pipeline, _entrypoint))) = &cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline.clone());
        }

        validate_cube_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let mut compiler = compiler(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        self.validate_shared(&compiled.repr)?;

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compiled.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let repr = compiled.repr.as_ref().map(|it| it.as_ref());

        let module_scopes = self.begin_pipeline_error_scopes(stream_id);
        let module_result =
            self.create_module(&compiled.entrypoint_name, repr, &compiled.source, mode);
        self.finish_pipeline_error_scopes(
            stream_id,
            module_scopes,
            Arc::from(format!(
                "while creating shader module for entrypoint `{}`",
                compiled.entrypoint_name
            )),
        )?;
        let module = module_result?;

        let pipeline_scopes = self.begin_pipeline_error_scopes(stream_id);
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.finish_pipeline_error_scopes(
            stream_id,
            pipeline_scopes,
            Arc::from(format!(
                "while creating bind-group/pipeline layouts and compute pipeline for entrypoint `{}`",
                compiled.entrypoint_name
            )),
        )?;
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = compiled.repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            let result = cache.insert(
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
            if let Err(err) = result {
                log::warn!("Unable to save the SPIR-V {err:?}");
            }
        }

        Ok(pipeline)
    }

    fn validate_shared(&self, repr: &Option<crate::AutoRepresentation>) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if let Some(shared_bytes) = shared_bytes
            && shared_bytes > max_smem
        {
            Err(ResourceLimitError::SharedMemory {
                requested: shared_bytes,
                max: max_smem,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        let reserved = stream.empty(size).unwrap();
        stream.mem_manage.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams.clone());

        // A read can consume resources produced on more than one logical stream. Their WebGPU
        // error scopes belong to those source streams, but the read future is the synchronization
        // boundary returned to the caller. Move every dependency's pending errors to the target
        // stream so a secondary mapping failure cannot hide the originating pipeline/task error.
        let mut dependency_errors = Vec::new();
        for source_stream_id in streams {
            if source_stream_id != stream_id {
                dependency_errors.push(
                    self.scheduler
                        .stream(&source_stream_id)
                        .take_pending_batch_errors(),
                );
            }
        }

        let stream = self.scheduler.stream(&stream_id);
        for errors in dependency_errors {
            stream.extend_pending_batch_errors(errors);
        }
        stream.read_resources(resources)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);

            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(r) => r,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, &[]);
        }
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let pipeline = match self.pipeline(kernel, &args, mode, &stream_id) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Launch(err));
                return;
            }
        };

        self.streams_pool.clear();
        args.buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));

        let resources = match self.prepare_bindings(args) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Io(err));
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, &self.streams_pool);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.end_profile(token)
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }
}

fn pipeline_creation_error(
    errors: Vec<(ErrorFilter, String)>,
    context: &str,
) -> Option<LaunchError> {
    if errors.is_empty() {
        return None;
    }

    let reason = errors
        .iter()
        .map(|(filter, message)| format!("WebGPU {filter:?} error {context}: {message}"))
        .collect::<Vec<_>>()
        .join("; ");

    if errors
        .iter()
        .any(|(filter, _)| *filter == ErrorFilter::OutOfMemory)
    {
        Some(LaunchError::OutOfMemory {
            reason,
            backtrace: BackTrace::capture(),
        })
    } else if errors
        .iter()
        .any(|(filter, _)| *filter == ErrorFilter::Validation)
    {
        Some(
            CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            }
            .into(),
        )
    } else {
        Some(LaunchError::Unknown {
            reason,
            backtrace: BackTrace::capture(),
        })
    }
}

fn compiler(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> AutoCompiler {
    let _ = options; // Unused without `spirv` feature
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan if options.supports_vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    #[test]
    fn wasm_pipeline_cache_invalidation_precedes_cache_lookup_correctness() {
        let source = include_str!("server.rs");
        let pipeline = source
            .split("    fn pipeline(")
            .nth(1)
            .expect("pipeline function must remain present")
            .split("    fn validate_shared(")
            .next()
            .expect("pipeline function must end before shared-memory validation");
        let invalidation = pipeline
            .find("self.pipeline_cache_invalidation.take()")
            .expect("Wasm pipeline retry must consume the invalidation marker");
        let lookup = pipeline
            .find("self.pipelines.get(&kernel_id)")
            .expect("pipeline cache lookup must remain present");

        assert!(invalidation < lookup);
        assert!(pipeline[invalidation..lookup].contains("self.pipelines.clear()"));
    }

    #[test]
    fn only_pipeline_creation_scopes_receive_cache_invalidation_marker_correctness() {
        let source = include_str!("server.rs");
        let finish = source
            .split("    fn finish_pipeline_error_scopes(")
            .nth(1)
            .expect("pipeline scope completion function must remain present")
            .split("    fn pipeline(")
            .next()
            .expect("pipeline scope completion must end before pipeline lookup");

        assert!(finish.contains("Some(self.pipeline_cache_invalidation.clone())"));
        assert!(finish.contains(
            ".finish_pipeline_error_scopes(scopes, context.clone(), pipeline_cache_invalidation)"
        ));
    }
}
