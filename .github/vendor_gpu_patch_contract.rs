const WGPU_API: &str = include_str!("../patches/wgpu-29.0.4/src/api/mod.rs");
const WGPU_QUEUE: &str = include_str!("../patches/wgpu-29.0.4/src/api/queue.rs");
const WGPU_DISPATCH: &str = include_str!("../patches/wgpu-29.0.4/src/dispatch.rs");
const WGPU_WEBGPU: &str = include_str!("../patches/wgpu-29.0.4/src/backend/webgpu.rs");
const BACKEND_BASE: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/backend/base.rs");
const BACKEND_WGSL: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/backend/wgsl.rs");
const BACKEND_METAL: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/backend/metal.rs");
const BACKEND_VULKAN: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/backend/vulkan.rs");
const VULKAN_FEATURES: &str =
    include_str!("../patches/cubecl-wgpu-0.10.0/src/backend/vulkan/features.rs");
const STREAM: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/compute/stream.rs");
const RUNTIME: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/runtime.rs");
const STORAGE: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/compute/storage.rs");
const STORAGE_RELEASE: &str =
    include_str!("../patches/cubecl-wgpu-0.10.0/src/compute/storage_release.rs");
const SERVER: &str = include_str!("../patches/cubecl-wgpu-0.10.0/src/compute/server.rs");
const BURN_CUBECL_PACKED_F16: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/kernel/packed_f16.rs");
const BURN_CUBECL_TENSOR_OPS: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/ops/tensor.rs");
const BURN_CUBECL_MODULE_OPS: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/ops/module.rs");
const BURN_CUBECL_QUANTIZED_OPS: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/ops/qtensor.rs");
const BURN_CUBECL_MATMUL_BASE: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/kernel/matmul/base.rs");
const BURN_CUBECL_QUANTIZED_MATMUL: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/kernel/matmul/quantized.rs");
const BURN_CUBECL_BASE_OPS: &str = include_str!("../patches/burn-cubecl-0.21.0/src/ops/base.rs");
const BURN_CUBECL_SELECT: &str =
    include_str!("../patches/burn-cubecl-0.21.0/src/kernel/index/select.rs");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("source contract start marker is missing: {start}"))
        .split(end)
        .next()
        .unwrap_or_else(|| panic!("source contract end marker is missing: {end}"))
}

#[test]
fn packed_quantized_uploads_honor_the_declared_axis_correctness() {
    let constructor = section(
        BURN_CUBECL_QUANTIZED_OPS,
        "fn new_quantized<R: CubeRuntime>(",
        "impl<R, F, I, BT> QTensorOps<Self>",
    );
    assert!(constructor.contains("QuantStore::PackedU32(packed_dim)"));
    assert!(constructor.contains(".checked_sub(packed_dim + 1)"));
    assert!(constructor.contains("shape_value[packed_axis] ="));
    assert!(!constructor.contains("shape_value[rank - 1] ="));

    let reshape = section(
        BURN_CUBECL_BASE_OPS,
        "pub fn q_reshape<R: CubeRuntime>(",
        "pub(crate) fn max_vector_size<R: CubeRuntime>",
    );
    assert!(reshape.contains("if curr_shape == &shape"));
    assert!(reshape.contains("return tensor;"));

    let select = section(
        BURN_CUBECL_SELECT,
        "fn quantized_select_rows_kernel<I: Numeric>(",
        "/// Select rows from a packed block-quantized matrix",
    );
    assert!(select.contains("values[source_row * values.stride(0)"));
    assert!(select.contains("scales[source_row * scales.stride(0)"));
    let route = section(
        BURN_CUBECL_QUANTIZED_OPS,
        "    fn q_select(",
        "    fn q_slice(",
    );
    assert!(route.contains("PackedU32(0)"));
    assert!(route.contains("kernel::select_quantized_rows"));
}

#[test]
fn packed_q4_matmul_uses_bounded_cooperative_path_with_portable_fallback_correctness() {
    let route = section(BURN_CUBECL_QUANTIZED_OPS, "    fn q_matmul(", "\n    }\n}");
    assert!(route.contains("MatmulStrategy::QuantizedPortable"));
    assert!(route.contains("use_packed_q4_rhs"));
    assert!(route.contains("TensorPrimitive::Float(lhs), TensorPrimitive::QFloat(rhs)"));
    assert!(route.contains("lhs.dtype == DType::F32"));
    assert!(route.contains("kernel::into_contiguous(lhs)"));
    assert!(route.contains("QuantValue::Q4S"));
    assert!(route.contains("QuantParam::F32"));
    assert!(route.contains("QuantStore::PackedU32(0 | 1)"));
    assert!(route.contains("MatmulStrategy::default()"));

    let launch = section(
        BURN_CUBECL_MATMUL_BASE,
        "        MatmulLaunch::QuantizedPortable => {",
        "\n        }\n    }\n\n    Ok(())",
    );
    let cooperative = launch
        .find("QuantizedCmmaAlgorithm")
        .expect("packed Q4 must try the bounded cooperative kernel");
    let unavailable = launch
        .find("if let Err(MatmulSetupError::Unavailable(_))")
        .expect("only unavailable cooperative support may select the fallback");
    let fallback = launch
        .find("QuantizedUnitAlgorithm")
        .expect("packed Q4 must retain the portable register fallback");
    assert!(cooperative < unavailable);
    assert!(unavailable < fallback);

    let cooperative_blueprint = section(
        BURN_CUBECL_QUANTIZED_MATMUL,
        "impl<RC, LL, RL, AL> Routine<RC> for QuantizedCmmaAlgorithm<LL, RL, AL>",
        "/// Resource-bounded register-tiled fallback for WebGPU and adapters without cooperative matrices.",
    );
    for marker in [
        "dtypes.lhs_stage = f16;",
        "dtypes.rhs_stage = f16;",
        "dtypes.lhs_register = f16;",
        "dtypes.rhs_register = f16;",
        "partition_k: Some(1)",
        "row_count: Some(quantized_cmma_row_count(",
        "PartitionBuffering::Single",
    ] {
        assert!(
            cooperative_blueprint.contains(marker),
            "bounded packed-Q4 cooperative kernel lost marker: {marker}"
        );
    }

    let fallback_blueprint = section(
        BURN_CUBECL_QUANTIZED_MATMUL,
        "impl<RC: RuntimeConfig> Routine<RC> for QuantizedUnitAlgorithm",
        "\n}",
    );
    assert!(fallback_blueprint.contains("SimpleMatmulFamily"));
    assert!(fallback_blueprint.contains("SyncFullCyclicLoading<ColMajorTilingOrder>"));
    assert!(fallback_blueprint.contains("SyncFullCyclicLoading<RowMajorTilingOrder>"));
    assert!(fallback_blueprint.contains("tile: TileSizeSelection::MinTileSize"));
    assert!(fallback_blueprint.contains("stage: StageScaling::Enabled(2)"));
    assert!(fallback_blueprint.contains("partition: PartitionScaling::Disabled"));
    assert!(fallback_blueprint.contains("quantized_unit_shared_memory_bytes"));
    assert!(fallback_blueprint.contains("max_shared_memory_size"));
    assert!(fallback_blueprint.contains("plane_dim: QUANTIZED_PORTABLE_UNIT_PLANE_DIM"));
    assert!(
        BURN_CUBECL_QUANTIZED_MATMUL.contains("const QUANTIZED_PORTABLE_UNIT_PLANE_DIM: u32 = 32;")
    );
}

#[test]
fn result_bearing_queue_api_is_public_and_forwarded_correctness() {
    assert!(WGPU_API.contains("pub use queue::*;"));

    let error = section(
        WGPU_QUEUE,
        "pub struct SubmittedWorkDoneError {",
        "/// Passed to [`Device::poll`]",
    );
    assert!(error.contains("reason: String"));
    assert!(error.contains("pub(crate) fn webgpu_promise_rejected(reason: String) -> Self"));
    assert!(error.contains("impl fmt::Display for SubmittedWorkDoneError"));
    assert!(error.contains("impl core::error::Error for SubmittedWorkDoneError"));

    let queue_api = section(
        WGPU_QUEUE,
        "    pub fn on_submitted_work_done_result(",
        "    /// Get the [`wgpu_hal`] device from this `Queue`.",
    );
    assert!(queue_api.contains("impl FnOnce(Result<(), SubmittedWorkDoneError>) + Send + 'static"));
    assert!(queue_api.contains("self.inner.on_submitted_work_done_result(Box::new(callback))"));
}

#[test]
fn dispatch_queue_result_default_preserves_existing_backends_correctness() {
    assert!(
        WGPU_DISPATCH.contains(
            "Box<dyn FnOnce(Result<(), crate::SubmittedWorkDoneError>) + Send + 'static>"
        )
    );
    assert!(
        WGPU_DISPATCH
            .contains("Box<dyn FnOnce(Result<(), crate::SubmittedWorkDoneError>) + 'static>")
    );

    let queue_interface = section(
        WGPU_DISPATCH,
        "pub trait QueueInterface: CommonTraits {",
        "pub trait ShaderModuleInterface: CommonTraits {",
    );
    let callback_api = queue_interface
        .find("fn on_submitted_work_done(&self, callback: BoxSubmittedWorkDoneCallback);")
        .expect("the existing callback API must remain the dispatch primitive");
    let result = queue_interface
        .find("fn on_submitted_work_done_result(&self, callback: BoxSubmittedWorkDoneResultCallback) {")
        .expect("dispatch must expose a result-bearing completion hook");
    assert!(callback_api < result);
    assert!(
        queue_interface[result..]
            .contains("self.on_submitted_work_done(Box::new(move || callback(Ok(()))));")
    );
}

#[test]
fn webgpu_queue_rejection_maps_to_public_error_correctness() {
    let queue = section(
        WGPU_WEBGPU,
        "impl dispatch::QueueInterface for WebQueue {",
        "impl Drop for WebQueue {",
    );
    let callback_api = queue
        .find("fn on_submitted_work_done(&self, callback: dispatch::BoxSubmittedWorkDoneCallback)")
        .expect("WebGPU must retain the original completion callback");
    let result = queue
        .find("fn on_submitted_work_done_result(")
        .expect("WebGPU must override the result-bearing completion hook");
    let promise = queue[result..]
        .find("let promise = self.inner.on_submitted_work_done();")
        .map(|offset| result + offset)
        .expect("WebGPU must observe its completion promise");
    let await_promise = queue[promise..]
        .find("wasm_bindgen_futures::JsFuture::from(promise).await")
        .map(|offset| promise + offset)
        .expect("WebGPU must await the completion promise");
    let success = queue[await_promise..]
        .find("Ok(_) => callback(Ok(()))")
        .map(|offset| await_promise + offset)
        .expect("resolved WebGPU completion must map to success");
    let rejection = queue[success..]
        .find("callback(Err(crate::SubmittedWorkDoneError::webgpu_promise_rejected(")
        .map(|offset| success + offset)
        .expect("rejected WebGPU completion must map to the public error");

    assert!(callback_api < result);
    assert!(result < promise);
    assert!(promise < await_promise);
    assert!(await_promise < success);
    assert!(success < rejection);
    assert!(queue[callback_api..result].contains("self.on_submitted_work_done_result"));
    assert!(queue[success..rejection].contains("dyn_ref::<js_sys::Error>()"));
    assert!(queue[success..rejection].contains(".or_else(|| error.as_string())"));
}

#[test]
fn upload_only_sync_submits_pending_writes_correctness() {
    assert!(STREAM.contains(
        "const fn stream_has_pending_submission(tasks_count: usize, pending_write_count: usize)"
    ));
    assert!(STREAM.contains("tasks_count != 0 || pending_write_count != 0"));

    let flush = section(STREAM, "    pub fn flush(", "    pub fn execute(");
    let pending_guard = flush
        .find("stream_has_pending_submission(self.tasks_count, self.pending_write_count)")
        .expect("flush must treat queued writes as pending work");
    let submission = flush
        .find("self.submit([tasks_encoder.finish()])")
        .expect("flush must submit an encoder for upload-only work");
    assert!(pending_guard < submission);
}

#[test]
fn sync_checks_queue_completion_and_drains_scoped_errors_correctness() {
    let sync = section(
        STREAM,
        "    pub fn sync(",
        "    /// Allocates a new empty buffer",
    );
    let completion = sync
        .find("queue.on_submitted_work_done_result")
        .expect("sync must observe result-bearing queue completion");
    let receive = sync
        .find("receiver.recv().await")
        .expect("sync must await queue completion");
    let drain = sync
        .find("for pending in pending_batch_errors")
        .expect("sync must drain queued WebGPU error scopes");
    let await_scope = sync
        .find("pending.future.await")
        .expect("sync must await each queued WebGPU error scope");
    let result = sync
        .find("sync_result(scoped_errors, flush_error, queue_completion_error)")
        .expect("sync must combine completion, scope, and flush failures");

    assert!(completion < receive);
    assert!(receive < drain);
    assert!(drain < await_scope);
    assert!(await_scope < result);
}

#[test]
fn cubecl_execution_capabilities_are_device_effective_correctness() {
    let create_server = section(
        RUNTIME,
        "pub(crate) fn create_server(",
        "/// Select the wgpu device and queue",
    );
    assert!(create_server.contains("let limits = setup.device.limits();"));
    assert!(create_server.contains("let features = setup.device.features();"));
    assert!(create_server.contains("timing_method_from_device_features(features)"));
    assert!(create_server.contains("features.contains(wgpu::Features::SUBGROUP)"));
    assert!(
        create_server
            .contains("backend::register_features(\n        &setup.adapter,\n        features,")
    );
    assert!(!create_server.contains("setup.adapter.limits()"));
    assert!(!create_server.contains("setup.adapter.features()"));
}

#[test]
fn cubecl_base_routes_effective_features_to_every_backend_correctness() {
    let registration = section(
        BACKEND_BASE,
        "pub fn register_features(\n",
        "#[cfg(feature = \"spirv\")]\npub fn register_vulkan_features(",
    );
    let vulkan = registration
        .find("register_vulkan_features(adapter, device_features")
        .expect("base registration must pass device features to Vulkan");
    let metal = registration
        .find("register_metal_features(adapter, device_features")
        .expect("base registration must pass device features to Metal");
    let wgsl = registration
        .find("wgsl::register_wgsl_features(device_features")
        .expect("base registration must pass device features to WGSL");
    assert!(vulkan < metal);
    assert!(metal < wgsl);

    let vulkan_route = section(
        BACKEND_BASE,
        "#[cfg(feature = \"spirv\")]\npub fn register_vulkan_features(",
        "#[cfg(not(feature = \"spirv\"))]",
    );
    assert!(vulkan_route.contains(
        "vulkan::register_vulkan_features(\n            adapter,\n            device_features,"
    ));

    let metal_route = section(
        BACKEND_BASE,
        "#[cfg(all(feature = \"msl\", target_os = \"macos\"))]\npub fn register_metal_features(",
        "#[cfg(not(all(feature = \"msl\", target_os = \"macos\")))]",
    );
    assert!(
        metal_route.contains(
            "metal::register_metal_features(adapter, device_features, props, comp_options)"
        )
    );
}

#[test]
fn wgsl_optional_capabilities_follow_effective_device_features_correctness() {
    let registration = section(
        BACKEND_WGSL,
        "pub fn register_wgsl_features(",
        "#[cfg(test)]",
    );
    assert!(registration.contains("register_types(props, device_features);"));
    assert!(registration.contains("if props.supports_type(ElemType::UInt(UIntKind::U64))"));

    let defaults = section(
        registration,
        "    let supported_types = [",
        "    let supported_atomic_types = [",
    );
    assert!(defaults.contains("ElemType::UInt(UIntKind::U32)"));
    assert!(defaults.contains("ElemType::Int(IntKind::I32)"));
    assert!(defaults.contains("ElemType::Float(FloatKind::F32)"));
    assert!(!defaults.contains("UIntKind::U64"));
    assert!(!defaults.contains("IntKind::I64"));
    assert!(!defaults.contains("FloatKind::F16"));
    assert!(!defaults.contains("FloatKind::F64"));

    let int64 = section(
        registration,
        "if device_features.contains(WgpuFeatures::SHADER_INT64) {",
        "if device_features.contains(WgpuFeatures::SHADER_F64) {",
    );
    assert!(int64.contains("UIntKind::U64"));
    assert!(int64.contains("IntKind::I64"));

    let f64 = section(
        registration,
        "if device_features.contains(WgpuFeatures::SHADER_F64) {",
        "if device_features.contains(WgpuFeatures::SHADER_F16) {",
    );
    assert!(f64.contains("FloatKind::F64"));

    let f16 = section(
        registration,
        "if device_features.contains(WgpuFeatures::SHADER_F16) {",
        "if device_features.contains(WgpuFeatures::SHADER_FLOAT32_ATOMIC) {",
    );
    assert!(f16.contains("FloatKind::F16"));

    let float32_atomic = section(
        registration,
        "if device_features.contains(WgpuFeatures::SHADER_FLOAT32_ATOMIC) {",
        "\n    }\n}",
    );
    assert!(float32_atomic.contains("StorageType::Atomic(ElemType::Float(FloatKind::F32))"));
}

#[test]
fn platform_registration_receives_and_gates_effective_features_correctness() {
    let metal_entry = section(
        BACKEND_METAL,
        "pub fn register_metal_features(",
        "fn register_features(",
    );
    assert!(metal_entry.contains("device_features: Features"));
    assert!(
        metal_entry.contains("register_features(&adapter, props, device_features, comp_options)")
    );

    let metal_registration = section(BACKEND_METAL, "fn register_features(", "fn register_cmma(");
    assert!(metal_registration.contains("register_types(props, features);"));
    for gate in [
        "features.contains(Features::EXPERIMENTAL_COOPERATIVE_MATRIX)",
        "features.contains(Features::SUBGROUP)",
        "features.contains(Features::SUBGROUP_BARRIER)",
        "device_features.contains(Features::SHADER_INT64)",
        "device_features.contains(Features::SHADER_F16)",
        "device_features.contains(Features::SHADER_FLOAT32_ATOMIC)",
    ] {
        assert!(
            metal_registration.contains(gate),
            "Metal capability lacks an effective device feature gate: {gate}"
        );
    }

    let vulkan_entry = section(
        BACKEND_VULKAN,
        "pub fn register_vulkan_features(",
        "/// Request device with required features",
    );
    assert!(vulkan_entry.contains("device_features: Features"));
    assert!(vulkan_entry.contains(
        "register_features(\n                &adapter,\n                props,\n                device_features,"
    ));

    let vulkan_registration = section(
        BACKEND_VULKAN,
        "/// Request device's supported features\nfn register_features(",
        "fn register_types(",
    );
    assert!(
        vulkan_registration
            .contains("ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features)")
    );
    assert!(!vulkan_registration.contains("adapter.features()"));

    let extension_selection = section(
        VULKAN_FEATURES,
        "    fn fill_extensions(&mut self, adapter: &vulkan::Adapter, features: Features) {",
        "    pub fn add_to_device_create(",
    );
    assert!(extension_selection.contains("adapter.required_device_extensions(features)"));
}

#[test]
fn wasm_storage_release_is_alias_safe_correctness() {
    let release = section(
        STORAGE_RELEASE,
        "pub(crate) fn release_resource_before_drop_if_unaliased",
        "#[cfg(test)]",
    );
    let sole_owner = release
        .find("Arc::strong_count(&resource_guard) == 1")
        .expect("explicit release must require sole ownership");
    let release_call = release
        .find("release(&resource)")
        .expect("sole-owner path must run the backend release operation");
    assert!(sole_owner < release_call);

    let wasm_release = section(
        STORAGE,
        "fn release_deallocated_buffer(buffer: wgpu::Buffer, resource_guard: Arc<()>)",
        "#[cfg(not(target_family = \"wasm\"))]",
    );
    assert!(wasm_release.contains("release_resource_before_drop_if_unaliased"));
    assert!(wasm_release.contains("resource_guard"));
    assert!(wasm_release.contains("wgpu::Buffer::destroy"));
}

#[test]
fn pipeline_invalidation_precedes_cache_lookup_correctness() {
    let pipeline = section(SERVER, "    fn pipeline(", "    fn validate_shared(");
    let invalidation = pipeline
        .find("self.pipeline_cache_invalidation.take()")
        .expect("pipeline lookup must consume asynchronous invalidation");
    let clear = pipeline[invalidation..]
        .find("self.pipelines.clear()")
        .map(|offset| invalidation + offset)
        .expect("pipeline invalidation must clear cached pipelines");
    let lookup = pipeline
        .find("self.pipelines.get(&kernel_id)")
        .expect("pipeline cache lookup must remain present");

    assert!(invalidation < clear);
    assert!(clear < lookup);
}

#[test]
fn packed_f16_unpack_kernels_never_require_shader_f16_correctness() {
    let unpack_route = section(
        BURN_CUBECL_PACKED_F16,
        "pub fn requires_packed_f16_unpack",
        "#[cube]\nfn widen_f16_bits_to_f32",
    );
    assert!(unpack_route.contains("tensor.dtype == DType::F16"));
    assert!(unpack_route.contains("!tensor"));
    assert!(unpack_route.contains("supports_type(ElemType::Float(FloatKind::F16))"));

    assert!(BURN_CUBECL_PACKED_F16.contains("packed: &Tensor<u32>"));
    assert!(BURN_CUBECL_PACKED_F16.contains("fn load_packed_f16"));
    assert!(BURN_CUBECL_PACKED_F16.contains("fn widen_f16_bits_to_f32"));
    assert!(BURN_CUBECL_PACKED_F16.contains("output: &mut Tensor<f32>"));
    assert!(
        BURN_CUBECL_PACKED_F16
            .contains("sums[row_lane * OUTPUT_COLUMNS_PER_UNIT + column_lane] +=")
    );
    assert!(BURN_CUBECL_PACKED_F16.contains("sums[column_lane] +="));
    assert!(!BURN_CUBECL_PACKED_F16.contains("FloatKind::F16) {"));
    assert!(!BURN_CUBECL_PACKED_F16.contains("convert_dtype(DType::F32)"));
}

#[test]
fn packed_f16_dispatch_covers_cast_linear_embedding_and_convolution_correctness() {
    let cast = section(
        BURN_CUBECL_TENSOR_OPS,
        "    fn float_cast(",
        "    fn float_unfold(",
    );
    assert!(cast.contains("dtype == FloatDType::F32"));
    assert!(cast.contains("requires_packed_f16_unpack(&tensor)"));
    assert!(cast.contains("return packed_f16_to_f32(tensor);"));

    let matmul = section(
        BURN_CUBECL_TENSOR_OPS,
        "    fn float_matmul(",
        "    fn float_cross(",
    );
    assert!(matmul.contains("lhs.dtype == DType::F32"));
    assert!(matmul.contains("requires_packed_f16_unpack(&rhs)"));
    assert!(matmul.contains("return packed_f16_rhs_matmul(lhs, rhs);"));

    let select = section(
        BURN_CUBECL_TENSOR_OPS,
        "    fn float_select(",
        "    fn float_select_add(",
    );
    assert!(select.contains("requires_packed_f16_unpack(&tensor)"));
    assert!(select.contains("return packed_f16_select_rows(tensor, dim, indices);"));

    let conv2d = section(
        BURN_CUBECL_MODULE_OPS,
        "    fn conv2d(",
        "    fn conv2d_x_backward(",
    );
    assert!(conv2d.contains("x.dtype == burn_backend::DType::F32"));
    assert!(conv2d.contains("requires_packed_f16_unpack(&weight)"));
    assert!(conv2d.contains("return packed_f16_conv2d(x, weight, bias, options);"));
}
