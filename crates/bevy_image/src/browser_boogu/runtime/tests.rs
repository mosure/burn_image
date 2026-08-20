
use std::collections::BTreeMap;

use burn_image::{
    ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_MANIFEST_SCHEMA_V2, ArtifactBundleId,
    ArtifactComponentId, ArtifactDependency, ArtifactFileRole, ArtifactProfileId, ArtifactSource,
    ModelId, NumericFormat, RemoteBaseUrl,
};

use super::*;

fn released_qwen_config() -> Qwen3VlConfig {
    Qwen3VlConfig::from_json(
            r#"{
              "text_config": {
                "vocab_size":151936,"hidden_size":4096,"intermediate_size":12288,
                "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
                "head_dim":128,"hidden_act":"silu","rms_norm_eps":1e-6,
                "max_position_embeddings":262144,"rope_theta":5000000,
                "rope_scaling":{"mrope_section":[24,20,20],"mrope_interleaved":true,"rope_type":"default"}
              },
              "vision_config": {
                "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
                "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
                "out_hidden_size":4096,"in_channels":3,"num_position_embeddings":2304,
                "deepstack_visual_indexes":[8,16,24],"hidden_act":"gelu_pytorch_tanh",
                "layer_norm_eps":1e-6
              },
              "tie_word_embeddings":false,"image_token_id":151655,"video_token_id":151656,
              "vision_start_token_id":151652,"vision_end_token_id":151653
            }"#,
        )
        .unwrap()
}

fn released_qwen_streaming_plan() -> Qwen3VlStreamingPlan {
    Qwen3VlStreamingPlan::released_f16(&released_qwen_config(), false).unwrap()
}

fn released_flux_vae_config() -> AutoencoderKlConfig {
    AutoencoderKlConfig::from_diffusers_json(
            r#"{
              "act_fn":"silu","block_out_channels":[128,256,512,512],
              "down_block_types":["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
              "force_upcast":true,"in_channels":3,"latent_channels":16,
              "layers_per_block":2,"mid_block_add_attention":true,"norm_num_groups":32,
              "out_channels":3,"sample_size":1024,"scaling_factor":0.3611,
              "shift_factor":0.1159,
              "up_block_types":["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
              "use_post_quant_conv":false,"use_quant_conv":false
            }"#,
        )
        .unwrap()
}

#[test]
fn parity_readback_ranges_bound_every_map_and_cover_exact_values_correctness() {
    let chunk_elements = BROWSER_PARITY_MAX_READBACK_CHUNK_BYTES / BROWSER_PARITY_F32_ELEMENT_BYTES;
    assert!(browser_parity_readback_ranges(0).is_empty());
    assert_eq!(
        browser_parity_readback_ranges(chunk_elements),
        vec![(0, chunk_elements)]
    );

    let elements = chunk_elements * 3 + 17;
    let ranges = browser_parity_readback_ranges(elements);
    assert_eq!(ranges.first(), Some(&(0, chunk_elements)));
    assert_eq!(ranges.last(), Some(&(chunk_elements * 3, elements)));
    assert_eq!(ranges.len(), 4);
    for (index, (start, end)) in ranges.iter().copied().enumerate() {
        assert!(start < end);
        assert!(end - start <= chunk_elements);
        if let Some((_, previous_end)) = index.checked_sub(1).map(|index| ranges[index]) {
            assert_eq!(start, previous_end);
        }
    }
}

fn tiny_manifest(bundle: &str, schema_version: u32) -> ArtifactManifest {
    let bytes = b"tiny";
    let mut manifest = ArtifactManifest {
        schema_version,
        bundle: ArtifactBundleId::new(bundle).unwrap(),
        profile: ArtifactProfileId::new("test-profile").unwrap(),
        model: ModelId::new(format!("test/{bundle}")).unwrap(),
        model_revision: "revision".into(),
        numeric_format: NumericFormat::F16,
        components: Vec::new(),
        files: vec![ArtifactFile {
            path: ArtifactPath::new("objects/tiny.bpk").unwrap(),
            size: bytes.len() as u64,
            sha256: Sha256Digest::calculate(bytes),
            role: ArtifactFileRole::Metadata,
            component: None,
            shard: None,
        }],
        dependencies: Vec::new(),
        metadata: BTreeMap::new(),
        content_digest: None,
    };
    manifest.seal().unwrap();
    manifest
}

fn dependency(role: &str, manifest: &ArtifactManifest) -> ArtifactDependency {
    ArtifactDependency {
        role: ArtifactComponentId::new(role).unwrap(),
        bundle: manifest.bundle.clone(),
        profile: manifest.profile.clone(),
        model: manifest.model.clone(),
        model_revision: manifest.model_revision.clone(),
        content_digest: manifest.content_digest.unwrap(),
    }
}

#[test]
fn browser_residency_selector_is_fail_closed_and_stably_labeled_correctness() {
    assert_eq!(
        BrowserBooguResidencyPolicy::default(),
        BrowserBooguResidencyPolicy::ResidentPackedQ4s
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("resident"),
        Some(BrowserBooguResidencyPolicy::ResidentPackedQ4s)
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("high-vram-resident-packed-f16"),
        Some(BrowserBooguResidencyPolicy::HighVramResidentPackedF16)
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("resident-q4"),
        Some(BrowserBooguResidencyPolicy::ResidentPackedQ4s)
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("high-vram-resident-dense-f32"),
        Some(BrowserBooguResidencyPolicy::HighVramResidentDenseF32)
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("qualification-f32"),
        Some(BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained)
    );
    assert_eq!(BrowserBooguResidencyPolicy::parse("low-vram"), None);
    assert_eq!(
        BrowserBooguResidencyPolicy::parse("low-vram-runtime-q8-denoiser"),
        Some(BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser)
    );
    assert_eq!(BrowserBooguResidencyPolicy::parse("streamed"), None);
    assert_eq!(
        BrowserBooguResidencyPolicy::parse(
            "low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
        ),
        Some(BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser)
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::HighVramResidentPackedF16.label(),
        "browser-high-vram-resident-packed-f16"
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::ResidentPackedQ4s.label(),
        "browser-resident-packed-q4s-block-up-to-128"
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::HighVramResidentDenseF32.label(),
        "browser-high-vram-resident-dense-f32"
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained.label(),
        "browser-qualification-per-request-f32-denoiser-retained"
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser.label(),
        "browser-low-vram-runtime-q8-denoiser"
    );
    assert_eq!(
        BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser.label(),
        "browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser"
    );
    assert_eq!(
        default_browser_low_vram_residency(BooguVariant::Image01Turbo),
        BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
    );
    for edit in [
        BooguVariant::Image01EditTurbo,
        BooguVariant::Image01EditTurbo1k5,
    ] {
        assert_eq!(
            default_browser_low_vram_residency(edit),
            BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser
        );
    }
}

#[test]
fn browser_public_variants_resident_q4_keep_all_modules_warm_correctness() {
    let q4_settings = q4_settings();
    for variant in [
        BooguVariant::Image01Turbo,
        BooguVariant::Image01EditTurbo,
        BooguVariant::Image01EditTurbo1k5,
    ] {
        assert!(BrowserExecutionPolicies::resident_packed_q4s(variant, &q4_settings).is_ok());
    }
    let policy =
        BrowserExecutionPolicies::resident_packed_q4s(BooguVariant::Image01Turbo, &q4_settings)
            .unwrap();
    assert_eq!(
        policy.residency,
        BrowserBooguResidencyPolicy::ResidentPackedQ4s
    );
    assert_eq!(
        policy.qwen_float,
        BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
    );
    assert_eq!(
        policy.vae_float,
        BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
    );
    assert_eq!(
        policy.denoiser_float,
        BooguFloatLoadPolicy::PackedQ4sWeightsF32Auxiliaries
    );
    assert_eq!(
        policy.denoiser_runtime_quantization,
        BooguDenoiserRuntimeQuantizationPolicy::Disabled
    );
    assert_eq!(
        policy.qwen_embedding_execution,
        Qwen3VlEmbeddingExecutionPolicy::DeviceRoutedChunks
    );
    assert!(policy.retain_qwen_stages);
    assert!(policy.retain_vae_stages);
    assert!(policy.retain_denoiser_stages);
    assert!(policy.eager_preload);
    assert!(policy.preload_denoiser_before_request);
    assert!(policy.defer_retained_qwen_synchronization);
    assert!(policy.defer_retained_denoiser_synchronization);
    assert!(policy.phase_boundary_memory_cleanup);
    assert!(!policy.release_unused_qwen_memory_after_stage);
    assert!(!policy.uses_packed_f16_denoiser_source());
    let dtypes = policy.execution_dtypes(BooguStorageProfile::Q4sBlockUpTo128F32);
    assert_eq!(dtypes.qwen_visual, DType::F32);
    assert_eq!(dtypes.vae, DType::F32);
    assert_eq!(dtypes.denoiser, DType::F32);
    assert_eq!(policy.vae_parameter_dtype(), DType::F16);
    assert_eq!(
        policy.denoiser_quantized_load_policy_report(),
        "preserve-stored-q4s-block-up-to-128-f32"
    );
    assert_eq!(
        policy.denoiser_quantized_linear_execution_policy_report(),
        "direct-quantized-matmul"
    );
    assert!(policy.packed_allocator_policy_is_exact());
    assert_eq!(
        policy.weight_traffic_contract(),
        "eager-preload/qwen+vae+denoiser/resident-q4s-matrices+embedding+packed-f16-convolutions+f32-auxiliaries/zero-inference-artifact-transfers/no-model-unload"
    );
        let source = include_str!("../runtime.rs");
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("browser runtime source has a production section");
    assert!(!production_source.contains(
        "resident packed-Q4S requires the canonical modular Qwen/VAE component manifests"
    ));
    assert_eq!(
        source.matches("self.report_resident_cache_audit(").count(),
        2
    );
    assert!(source.contains("resident_weights_preserved"));
    assert!(production_source.contains("decoder.decoder_float_dtype()"));
    assert!(!production_source.contains("let loaded_dtype: DType = decoder.float_dtype()"));
    assert!(
        BrowserExecutionPolicies::resident_packed_q4s(
            BooguVariant::Image01Turbo,
            &mixed_f16_settings(),
        )
        .is_err()
    );
}

#[test]
fn browser_turbo_low_vram_preloads_packed_f16_for_dense_f32_stages_correctness() {
    let policy = BrowserExecutionPolicies::low_vram_preloaded_packed_f16_denoiser(
        BooguVariant::Image01Turbo,
        &mixed_f16_settings(),
    )
    .unwrap();
    assert_eq!(
        policy.residency,
        BrowserBooguResidencyPolicy::LowVramPreloadedPackedF16Denoiser
    );
    assert_eq!(
        policy.denoiser_quantized,
        BooguQuantizedLoadPolicy::Preserve
    );
    assert_eq!(
        policy.denoiser_runtime_quantization,
        BooguDenoiserRuntimeQuantizationPolicy::Disabled
    );
    assert_eq!(
        policy.denoiser_retaining_wrapper_adapter,
        BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul
    );
    assert!(!policy.retain_denoiser_stages);
    assert!(policy.uses_packed_f16_denoiser_source());
    assert_eq!(
        policy.denoiser_execution_kind,
        BrowserDenoiserExecutionKind::PackedF16DeviceWidenDenseF32
    );
    assert_eq!(
        policy.denoiser_quantized_linear_execution_policy_report(),
        "not-applicable-packed-f16-storage"
    );
    assert_eq!(
        policy.denoiser_quantized_load_policy_report(),
        "not-applicable-packed-f16-storage"
    );
    assert_eq!(
        policy.denoiser_storage_policy(),
        "authenticated-compact-f16/padded-u32-retained/dense-f32-per-semantic-stage"
    );
    assert_eq!(
        policy.denoiser_linear_execution_policy(),
        "packed-f16-storage/device-widen-f32-per-semantic-stage/dense-f32-matmul"
    );
    assert!(policy.preload_denoiser_before_request);
    assert!(policy.requires_packed_f16_request_preload());
    assert!(!policy.eager_preload);
    assert!(!policy.defer_retained_denoiser_synchronization);
    assert!(!policy.defer_retained_qwen_synchronization);
    assert!(!policy.retain_qwen_stages);
    assert!(!policy.release_unused_qwen_memory_after_stage);
    assert!(policy.packed_qwen_instruction_handoff);
    assert_eq!(
        policy.qwen_text_block_load_synchronization,
        Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward
    );
    assert!(policy.packed_allocator_policy_is_exact());
    assert_eq!(
        policy.packed_qwen_instruction_handoff_policy(),
        BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY
    );
    assert!(!policy.request_scoped_surface_acquire_suspended);
    assert!(!policy.require_persistent_range_cache);
    assert!(
        !policy
            .provenance_backend()
            .contains(BROWSER_SURFACE_INFERENCE_PROVENANCE_SUFFIX)
    );
    let mut unsafe_qwen_release = policy;
    unsafe_qwen_release.release_unused_qwen_memory_after_stage = true;
    assert!(!unsafe_qwen_release.packed_allocator_policy_is_exact());
    let mut missing_phase_cleanup = policy;
    missing_phase_cleanup.packed_qwen_instruction_handoff = false;
    assert!(!missing_phase_cleanup.packed_allocator_policy_is_exact());
    let mut deferred_qwen = policy;
    deferred_qwen.defer_retained_qwen_synchronization = true;
    assert!(!deferred_qwen.packed_allocator_policy_is_exact());
    let mut retained_qwen = policy;
    retained_qwen.retain_qwen_stages = true;
    assert!(!retained_qwen.packed_allocator_policy_is_exact());
    let mut missing_pre_forward_barrier = policy;
    missing_pre_forward_barrier.qwen_text_block_load_synchronization =
        Qwen3VlTextBlockLoadSynchronizationPolicy::PostForwardOnly;
    assert!(!missing_pre_forward_barrier.packed_allocator_policy_is_exact());
    let mut submission_policy_drift = policy;
    submission_policy_drift.qwen_text_layer_submission_policy =
        BROWSER_DEFAULT_QWEN_TEXT_LAYER_SUBMISSION_POLICY;
    assert!(!submission_policy_drift.packed_allocator_policy_is_exact());
    let ordinary = policy.for_ordinary_browser_factory(true);
    assert!(ordinary.require_persistent_range_cache);
    assert!(ordinary.request_scoped_surface_acquire_suspended);
    assert_eq!(
        ordinary.weight_traffic_contract(),
        "persistent-transport-part-cache/qwen+vae+packed-f16-denoiser-rehydrated-before-each-request/zero-dmd-artifact-transfers/zero-repeat-network-required/request-scoped-packed-cache-evicted-before-vae/dense-f32-materialized-per-semantic-stage"
    );
    assert_eq!(
        ordinary.packed_f16_dmd_vae_handoff_policy(),
        BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY
    );
    assert_eq!(
        ordinary.provenance_backend(),
        "burn-webgpu/browser-low-vram-preloaded-packed-f16-dense-f32-per-stage-denoiser/request-scoped-packed-cache-evicted-before-vae/request-scoped-surface-acquire-suspended"
    );
    assert!(
        BrowserExecutionPolicies::low_vram_preloaded_packed_f16_denoiser(
            BooguVariant::Image01EditTurbo,
            &mixed_f16_settings(),
        )
        .is_err()
    );
}

#[test]
fn browser_resident_policy_keeps_selected_pipeline_warm_between_requests_correctness() {
    let packed = BrowserExecutionPolicies::resident_packed_f16(&mixed_f16_settings())
        .unwrap()
        .for_ordinary_browser_factory(false);
    assert_eq!(
        packed.residency,
        BrowserBooguResidencyPolicy::HighVramResidentPackedF16
    );
    assert_eq!(
        packed.qwen_float,
        BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
    );
    assert_eq!(
        packed.vae_float,
        BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
    );
    assert_eq!(
        packed.denoiser_float,
        BooguFloatLoadPolicy::PackedF16WeightsF32Auxiliaries
    );
    assert!(packed.eager_preload);
    assert!(packed.retain_qwen_stages);
    assert!(packed.retain_vae_stages);
    assert!(packed.retain_denoiser_stages);
    assert_eq!(
        packed.weight_traffic_contract(),
        "eager-preload/qwen+vae+denoiser/resident-f16-weights/fused-f32-accumulate/zero-inference-artifact-transfers/zero-full-stage-widening"
    );

    let policy = BrowserExecutionPolicies::resident_dense_f32(&mixed_f16_settings())
        .unwrap()
        .for_ordinary_browser_factory(false);
    assert_eq!(
        policy.residency,
        BrowserBooguResidencyPolicy::HighVramResidentDenseF32
    );
    assert!(policy.eager_preload);
    assert!(policy.retain_qwen_stages);
    assert!(policy.retain_vae_stages);
    assert!(policy.retain_denoiser_stages);
    assert!(policy.defer_retained_qwen_synchronization);
    assert!(policy.defer_retained_denoiser_synchronization);
    assert!(!policy.release_unused_qwen_memory_after_stage);
    assert!(!policy.uses_packed_f16_denoiser_source());
    assert!(!policy.requires_packed_f16_request_preload());
    assert!(!policy.request_scoped_surface_acquire_suspended);
    assert!(policy.require_persistent_range_cache);
    assert_eq!(
        policy.weight_traffic_contract(),
        "eager-preload/qwen+vae+denoiser/zero-inference-artifact-transfers"
    );
}

#[test]
fn browser_turbo_packed_f16_resource_and_traffic_plan_is_exact_correctness() {
    let plan = validate_browser_packed_f16_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .unwrap();
    assert_eq!(
        plan.qwen_text_layer_allocation_policy,
        Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent.label()
    );
    assert_eq!(
        plan.qwen_text_block_load_synchronization_policy,
        Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward.label()
    );
    assert_eq!(
        plan.qwen_text_layer_submission_policy,
        BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY
    );
    assert!(plan.qwen_text_layer_persistent_pool_requires_measured_gpu_gate);
    assert_eq!(plan.authenticated_artifact_bytes, 19_870_166_528);
    assert_eq!(plan.canonical_compact_f16_payload_bytes, 19_869_996_096);
    assert_eq!(plan.retained_packed_f16_denoiser_bytes, 19_870_010_624);
    assert_eq!(plan.inserted_padding_elements, 7_264);
    assert_eq!(plan.padded_f16_elements, 9_935_005_312);
    assert_eq!(plan.expected_stage_count, 46);
    assert_eq!(plan.expected_object_count, 106);
    assert_eq!(plan.expected_tensor_count, 912);
    assert_eq!(plan.max_packed_stage_bytes, 876_827_328);
    assert_eq!(plan.max_materialized_stage_f32_bytes, 1_753_654_656);
    assert_eq!(plan.max_packed_object_bytes, 254_251_904);
    assert_eq!(plan.max_materialized_object_f32_bytes, 508_503_808);
    assert_eq!(plan.materialized_f32_bytes_per_dmd_step, 39_740_021_248);
    assert_eq!(plan.preload_workspace_bytes, 2_434_252_800);
    assert_eq!(plan.preload_peak_bytes, 22_304_263_424);
    assert_eq!(plan.activation_reserve_bytes, 4_868_505_600);
    assert_eq!(plan.conservative_planned_device_bytes, 26_492_170_880);
    assert_eq!(plan.strict_device_cap_bytes, 32_000_000_000);
    assert_eq!(plan.expected_stage_materializations_per_request, 184);
    assert_eq!(plan.expected_object_unpacks_per_request, 424);
    assert_eq!(plan.expected_packed_read_bytes_per_request, 79_480_042_496);
    assert_eq!(plan.expected_f32_write_bytes_per_request, 158_960_084_992);
    assert!(!plan.on_device_quantized_execution_claimed);
    let event = serde_json::to_value(browser_packed_f16_resource_plan_event(plan)).unwrap();
    assert_eq!(event["event"], "packed_f16_resource_plan");
    assert_eq!(event["authenticated_artifact_bytes"], 19_870_166_528_u64);
    assert_eq!(
        event["retained_packed_f16_denoiser_bytes"],
        19_870_010_624_u64
    );
    assert_eq!(event["expected_stage_materializations_per_request"], 184);
    assert_eq!(event["expected_object_unpacks_per_request"], 424);
    assert_eq!(
        event["expected_packed_read_bytes_per_request"],
        79_480_042_496_u64
    );
    assert_eq!(
        event["expected_f32_write_bytes_per_request"],
        158_960_084_992_u64
    );
    assert_eq!(event["on_device_quantized_execution_claimed"], false);
    assert!(
        validate_browser_packed_f16_resource_plan_with_cap(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            26_492_170_881,
        )
        .is_ok()
    );
    assert!(
        validate_browser_packed_f16_resource_plan_with_cap(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
            26_492_170_880,
        )
        .is_err()
    );
}

#[test]
fn browser_turbo_packed_f16_lifecycle_requires_complete_cache_and_zero_dmd_io_correctness() {
    let plan = validate_browser_packed_f16_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .unwrap();
    let exact = BrowserPackedF16DenoiserLifecycleReport {
        cache_state: "ready",
        cache_ready: true,
        cached_stages: 46,
        cached_objects: 106,
        cached_tensors: 912,
        cached_bytes: 19_870_010_624,
        authenticated_artifact_bytes: 19_870_166_528,
        packed_upload_bytes: 19_870_010_624,
        stage_materializations: 184,
        object_unpacks: 424,
        packed_read_bytes: 79_480_042_496,
        f32_write_bytes: 158_960_084_992,
        preload_attempt_count: 1,
        failure_count: 0,
        dmd_artifact_traffic: BrowserArtifactTrafficReport::default(),
        synchronization_pending: false,
        matches_plan: false,
    };
    assert!(
        validate_packed_f16_denoiser_lifecycle(BooguVariant::Image01Turbo, plan, 4, exact)
            .unwrap()
            .matches_plan
    );
    let mut second_request = exact;
    second_request.authenticated_artifact_bytes = 39_740_333_056;
    second_request.packed_upload_bytes = 39_740_021_248;
    second_request.preload_attempt_count = 2;
    assert!(
            validate_packed_f16_denoiser_lifecycle(
                BooguVariant::Image01Turbo,
                plan,
                4,
                second_request,
            )
            .unwrap()
            .matches_plan
        );
    let mut partial = exact;
    partial.cached_objects -= 1;
    assert!(
        validate_packed_f16_denoiser_lifecycle(BooguVariant::Image01Turbo, plan, 4, partial)
            .is_err()
    );
    let mut reloaded = exact;
    reloaded.dmd_artifact_traffic.object_reads = 1;
    assert!(
        validate_packed_f16_denoiser_lifecycle(BooguVariant::Image01Turbo, plan, 4, reloaded)
            .is_err()
    );
}

#[test]
fn browser_turbo_dmd_vae_handoff_requires_exact_latent_and_empty_cache_correctness() {
    let plan = validate_browser_packed_f16_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .unwrap();
    let digest = Sha256Digest::calculate(b"exact-final-dmd-latent");
    let ready = BrowserPackedF16CacheEvidence {
        state: "ready".into(),
        cache_ready: true,
        cached_stages: 46,
        cached_objects: 106,
        cached_tensors: 912,
        cached_bytes: 19_870_010_624,
    };
    let empty = BrowserPackedF16CacheEvidence {
        state: "empty".into(),
        cache_ready: false,
        cached_stages: 0,
        cached_objects: 0,
        cached_tensors: 0,
        cached_bytes: 0,
    };
    let exact = BrowserPackedF16DmdVaeHandoffReport {
        policy: BROWSER_PACKED_F16_DMD_VAE_HANDOFF_POLICY.into(),
        next_request_rehydration_policy: BROWSER_PACKED_F16_NEXT_REQUEST_REHYDRATION_POLICY.into(),
        shape: vec![1, 16, 128, 128],
        dtype: "f32".into(),
        element_count: 262_144,
        payload_bytes: 1_048_576,
        device_to_host_readback_bytes: 2_097_152,
        host_to_device_upload_bytes: 1_048_576,
        total_transfer_bytes: 3_145_728,
        before_sha256: digest,
        after_sha256: digest,
        all_finite: true,
        not_all_zero: true,
        digest_matches: true,
        wrapper_cached_stages_before_clear: 0,
        wrapper_cached_stages_after_clear: 0,
        synchronization_pending_before_cleanup: false,
        synchronization_pending_after_cleanup: false,
        rope_cache_cleared: true,
        cleanup_completed: true,
        packed_cache_before_cleanup: ready,
        packed_cache_after_cleanup: empty,
        preload_attempt_count: 1,
        expected_next_request_preload_attempt_count: 2,
    };
    validate_packed_f16_dmd_vae_handoff_report(
        BooguVariant::Image01Turbo,
        plan,
        [1, 16, 128, 128],
        &exact,
    )
    .unwrap();
    let event = serde_json::to_value(browser_packed_f16_dmd_vae_handoff_event(
        RunId(31),
        exact.clone(),
    ))
    .unwrap();
    assert_eq!(event["event"], "packed_f16_dmd_vae_handoff");
    assert_eq!(event["run_id"], 31);
    assert_eq!(event["report"], serde_json::to_value(&exact).unwrap());

    for invalid in [
        {
            let mut invalid = exact.clone();
            invalid.packed_cache_after_cleanup.cached_bytes = 4;
            invalid
        },
        {
            let mut invalid = exact.clone();
            invalid.after_sha256 = Sha256Digest::calculate(b"mutated");
            invalid
        },
        {
            let mut invalid = exact.clone();
            invalid.all_finite = false;
            invalid
        },
        {
            let mut invalid = exact.clone();
            invalid.wrapper_cached_stages_after_clear = 1;
            invalid
        },
        {
            let mut invalid = exact.clone();
            invalid.expected_next_request_preload_attempt_count = 3;
            invalid
        },
    ] {
        assert!(
            validate_packed_f16_dmd_vae_handoff_report(
                BooguVariant::Image01Turbo,
                plan,
                [1, 16, 128, 128],
                &invalid,
            )
            .is_err()
        );
    }
}

#[test]
fn browser_turbo_packed_f16_lifecycle_event_preserves_exact_report_json_correctness() {
    let lifecycle = BrowserPackedF16DenoiserLifecycleReport {
        cache_state: "ready",
        cache_ready: true,
        cached_stages: 46,
        cached_objects: 106,
        cached_tensors: 912,
        cached_bytes: 19_870_010_624,
        authenticated_artifact_bytes: 19_870_166_528,
        packed_upload_bytes: 19_870_010_624,
        stage_materializations: 184,
        object_unpacks: 424,
        packed_read_bytes: 79_480_042_496,
        f32_write_bytes: 158_960_084_992,
        preload_attempt_count: 1,
        failure_count: 0,
        dmd_artifact_traffic: BrowserArtifactTrafficReport::default(),
        synchronization_pending: false,
        matches_plan: true,
    };
    let event =
        serde_json::to_value(browser_packed_f16_denoiser_lifecycle_event(lifecycle)).unwrap();
    assert_eq!(
        event,
        serde_json::json!({
            "event": "packed_f16_denoiser_lifecycle",
            "lifecycle": {
                "cache_state": "ready",
                "cache_ready": true,
                "cached_stages": 46,
                "cached_objects": 106,
                "cached_tensors": 912,
                "cached_bytes": 19_870_010_624_u64,
                "authenticated_artifact_bytes": 19_870_166_528_u64,
                "packed_upload_bytes": 19_870_010_624_u64,
                "stage_materializations": 184,
                "object_unpacks": 424,
                "packed_read_bytes": 79_480_042_496_u64,
                "f32_write_bytes": 158_960_084_992_u64,
                "preload_attempt_count": 1,
                "failure_count": 0,
                "dmd_artifact_traffic": {
                    "object_reads": 0,
                    "object_read_bytes": 0,
                    "range_reads": 0,
                    "range_read_bytes": 0,
                    "verified_objects": 0,
                    "cache_lookups": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_read_bytes": 0,
                    "network_requests": 0,
                    "network_response_bytes": 0,
                    "cache_writes": 0,
                    "cache_write_bytes": 0,
                    "cache_evictions": 0,
                    "cache_evicted_entries": 0,
                    "cache_invalid_entries": 0,
                    "integrity_refetches": 0
                },
                "synchronization_pending": false,
                "matches_plan": true
            }
        })
    );
    assert_eq!(event["lifecycle"], serde_json::to_value(lifecycle).unwrap());
}

#[test]
fn browser_turbo_pre_dmd_input_event_has_exact_diagnostic_provenance_correctness() {
    let tensor = |name: &str, shape: Vec<usize>, element_count: usize| {
        BrowserPackedF16TensorInputDiagnostic {
            name: name.into(),
            shape,
            dtype: "f32".into(),
            element_count,
            finite_element_count: element_count,
            all_finite: true,
            max_abs: Some(4.0),
            mean: Some(0.25),
            rms: Some(1.0),
            sha256: Sha256Digest::calculate(name.as_bytes()),
        }
    };
    let diagnostics = BrowserPackedF16PreDmdInputDiagnostics {
        scope: "rendered-model-smoke/ordinary-turbo-packed-f16/pre-dmd-input-readback".into(),
        policy: BrowserPackedF16PreDmdPolicyEvidence {
            qwen_release_unused_memory_after_stage: false,
            qwen_text_block_load_synchronization_policy:
                Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward
                    .label()
                    .into(),
            qwen_text_layer_submission_policy: BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY
                .into(),
            packed_qwen_instruction_handoff_policy: BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY.into(),
            cleanup_completed: true,
            post_cleanup_packed_cache: BrowserPackedF16CacheEvidence {
                state: "ready".into(),
                cache_ready: true,
                cached_stages: 46,
                cached_objects: 106,
                cached_tensors: 912,
                cached_bytes: 19_870_010_624,
            },
        },
        dmd_steps: 4,
        instruction: tensor("instruction", vec![1, 45, 4096], 184_320),
        initial_latent: tensor("initial_latent", vec![1, 16, 128, 128], 262_144),
        renoise: (0..3)
            .map(|index| tensor(&format!("renoise_{index}"), vec![1, 16, 128, 128], 262_144))
            .collect(),
        first_timestep: tensor("first_timestep", vec![1], 1),
        all_inputs_finite: true,
    };
    let event = serde_json::to_value(browser_packed_f16_pre_dmd_input_diagnostics_event(
        RunId(17),
        diagnostics.clone(),
    ))
    .unwrap();
    assert_eq!(event["event"], "packed_f16_pre_dmd_input_diagnostics");
    assert_eq!(event["run_id"], 17);
    assert_eq!(
        event["diagnostics"],
        serde_json::to_value(&diagnostics).unwrap()
    );
    assert_eq!(event["diagnostics"]["dmd_steps"], 4);
    assert_eq!(event["diagnostics"]["renoise"].as_array().unwrap().len(), 3);
    assert_eq!(
        event["diagnostics"]["policy"]["packed_qwen_instruction_handoff_policy"],
        BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY
    );
    assert_eq!(
        event["diagnostics"]["policy"]["post_cleanup_packed_cache"]["cached_bytes"],
        19_870_010_624_u64
    );
}

#[test]
fn browser_turbo_block0_boundary_report_is_prefix_fail_closed_correctness() {
    let tensor = |boundary: Qwen3VlTextLayerDiagnosticBoundary, sha256: Sha256Digest| {
        let shape = if boundary == Qwen3VlTextLayerDiagnosticBoundary::InputLayerNormGamma {
            vec![4096]
        } else {
            vec![1, 45, 4096]
        };
        let element_count = shape.iter().product();
        BrowserPackedF16TensorInputDiagnostic {
            name: format!(
                "qwen_text_block_00_{}_immediate_post_sync",
                boundary.label()
            ),
            shape,
            dtype: "f32".into(),
            element_count,
            finite_element_count: element_count,
            all_finite: true,
            max_abs: Some(4.0),
            mean: Some(0.25),
            rms: Some(1.0),
            sha256,
        }
    };
    let input_sha = Sha256Digest::calculate(b"layer-input");
    let control = BrowserQwenInstructionDiagnosticControl::default();
    let mut complete = None;
    for boundary in BROWSER_PACKED_F16_QWEN_BLOCK0_BOUNDARIES {
        let sha = if matches!(
            boundary,
            Qwen3VlTextLayerDiagnosticBoundary::LayerInput
                | Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary
        ) {
            input_sha
        } else {
            Sha256Digest::calculate(boundary.label().as_bytes())
        };
        let outcome = control
            .record_block0_boundary(
                Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent,
                Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward,
                BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
                boundary,
                tensor(boundary, sha),
            )
            .unwrap();
        assert!(outcome.failure_message.is_none());
        if outcome.report.is_some() {
            assert_eq!(
                boundary,
                Qwen3VlTextLayerDiagnosticBoundary::FinalResidualOutput
            );
            complete = outcome.report;
        }
    }
    let complete = complete.unwrap();
    assert!(complete.complete);
    assert_eq!(complete.captured_boundary_count, 9);
    assert_eq!(complete.identity_add_canary_matches_input, Some(true));
    let event = serde_json::to_value(browser_packed_f16_qwen_block0_execution_diagnostics_event(
        RunId(23),
        complete,
    ))
    .unwrap();
    assert_eq!(
        event["event"],
        "packed_f16_qwen_block0_execution_diagnostics"
    );
    assert_eq!(event["run_id"], 23);
    assert_eq!(
        event["diagnostics"]["boundaries"].as_array().unwrap().len(),
        9
    );

    let failing = BrowserQwenInstructionDiagnosticControl::default();
    for boundary in [
        Qwen3VlTextLayerDiagnosticBoundary::LayerInput,
        Qwen3VlTextLayerDiagnosticBoundary::InputLayerNormGamma,
    ] {
        failing
            .record_block0_boundary(
                Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent,
                Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward,
                BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
                boundary,
                tensor(boundary, input_sha),
            )
            .unwrap();
    }
    let failure = failing
        .record_block0_boundary(
            Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent,
            Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward,
            BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY,
            Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary,
            tensor(
                Qwen3VlTextLayerDiagnosticBoundary::IdentityAddCanary,
                Sha256Digest::calculate(b"alias-drift"),
            ),
        )
        .unwrap();
    assert!(failure.failure_message.is_some());
    let report = failure.report.unwrap();
    assert!(!report.complete);
    assert_eq!(report.captured_boundary_count, 3);
    assert_eq!(report.identity_add_canary_matches_input, Some(false));
    assert_eq!(
        report.first_failure_boundary.as_deref(),
        Some("identity_add_canary")
    );
    assert_eq!(
        report.failure_reason.as_deref(),
        Some("identity-add-canary-mismatch")
    );
}

#[test]
fn browser_turbo_qwen_handoff_events_and_transfer_accounting_are_exact_correctness() {
    let tensor = |name: String, sha256: Sha256Digest| BrowserPackedF16TensorInputDiagnostic {
        name,
        shape: vec![1, 45, 4096],
        dtype: "f32".into(),
        element_count: 184_320,
        finite_element_count: 184_320,
        all_finite: true,
        max_abs: Some(24.5),
        mean: Some(0.125),
        rms: Some(1.0),
        sha256,
    };
    let mut stages = vec![tensor(
        "qwen_embedding_output".into(),
        Sha256Digest::calculate(b"embedding"),
    )];
    stages.extend((0..36).map(|index| {
        tensor(
            format!("qwen_text_block_{index:02}_output"),
            Sha256Digest::calculate(format!("block-{index}").as_bytes()),
        )
    }));
    let final_sha = Sha256Digest::calculate(b"final-norm");
    stages.push(tensor("qwen_final_norm_output".into(), final_sha));
    assert!(packed_f16_qwen_stage_diagnostic_names_are_exact(
        &stages, 38
    ));

    let pre_handoff_sha = Sha256Digest::calculate(b"trimmed-instruction");
    let block_00_immediate_post_sync = BrowserPackedF16QwenBlock0PostSyncDiagnostic {
        scope: BROWSER_PACKED_F16_QWEN_BLOCK0_POST_SYNC_SCOPE.into(),
        block0_execution_mode: BROWSER_QWEN_BLOCK0_SERIALIZED_DIAGNOSTIC_MODE.into(),
        text_layer_allocation_policy: Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent
            .label()
            .into(),
        text_block_load_synchronization_policy:
            Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward
                .label()
                .into(),
        qwen_text_layer_submission_policy: BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY
            .into(),
        tensor: tensor(
            "qwen_text_block_00_output".into(),
            Sha256Digest::calculate(b"block-0"),
        ),
        all_finite: true,
        not_all_zero: true,
    };
    let pre = BrowserPackedF16QwenPreHandoffDiagnostics {
        scope: BROWSER_PACKED_F16_QWEN_PRE_HANDOFF_SCOPE.into(),
        effective_instruction_length: 45,
        expected_stage_output_count: 38,
        stage_outputs: stages,
        stage_names_exact: true,
        qwen_last_hidden_state_before_trim: tensor(
            "qwen_last_hidden_state_before_trim".into(),
            final_sha,
        ),
        instruction_after_trim_cast_before_handoff: tensor(
            "instruction_after_trim_cast_before_handoff".into(),
            pre_handoff_sha,
        ),
        all_tensors_finite: true,
        no_tensor_all_zero: true,
        first_non_finite_tensor: None,
        first_all_zero_tensor: None,
        final_norm_matches_returned_output: true,
        block_00_immediate_post_sync,
        block_00_immediate_matches_delayed_capture: true,
    };
    let pre_event = serde_json::to_value(browser_packed_f16_qwen_pre_handoff_diagnostics_event(
        RunId(19),
        pre,
    ))
    .unwrap();
    assert_eq!(
        pre_event["event"],
        "packed_f16_qwen_pre_handoff_diagnostics"
    );
    assert_eq!(pre_event["run_id"], 19);
    assert_eq!(
        pre_event["diagnostics"]["stage_outputs"]
            .as_array()
            .unwrap()
            .len(),
        38
    );

    let cache = BrowserPackedF16CacheEvidence {
        state: "ready".into(),
        cache_ready: true,
        cached_stages: 46,
        cached_objects: 106,
        cached_tensors: 912,
        cached_bytes: 19_870_010_624,
    };
    let after = tensor("instruction_after_handoff".into(), pre_handoff_sha);
    let report = BrowserPackedF16QwenInstructionHandoffReport {
        policy: BROWSER_PACKED_F16_QWEN_HANDOFF_POLICY.into(),
        qwen_release_unused_memory_after_stage: false,
        qwen_text_layer_allocation_policy: Qwen3VlTextLayerAllocationPolicy::ExactSizePersistent
            .label()
            .into(),
        qwen_text_block_load_synchronization_policy:
            Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward
                .label()
                .into(),
        qwen_text_layer_submission_policy: BROWSER_PACKED_F16_QWEN_TEXT_LAYER_SUBMISSION_POLICY
            .into(),
        shape: after.shape.clone(),
        dtype: after.dtype.clone(),
        element_count: after.element_count,
        payload_bytes: 737_280,
        device_to_host_readback_bytes: 1_474_560,
        host_to_device_upload_bytes: 737_280,
        total_transfer_bytes: 2_211_840,
        before_sha256: pre_handoff_sha,
        after_sha256: pre_handoff_sha,
        all_finite: true,
        not_all_zero: true,
        digest_matches: true,
        cleanup_completed: true,
        packed_cache: cache.clone(),
    };
    let post = BrowserPackedF16QwenPostHandoffDiagnostics {
        scope: BROWSER_PACKED_F16_QWEN_POST_HANDOFF_SCOPE.into(),
        handoff: report.clone(),
        instruction_after_handoff: after,
    };
    let post_event = serde_json::to_value(browser_packed_f16_qwen_post_handoff_diagnostics_event(
        RunId(19),
        post,
    ))
    .unwrap();
    assert_eq!(
        post_event["event"],
        "packed_f16_qwen_post_handoff_diagnostics"
    );
    assert_eq!(post_event["diagnostics"]["handoff"]["digest_matches"], true);

    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["payload_bytes"], 737_280);
    assert_eq!(report["device_to_host_readback_bytes"], 1_474_560);
    assert_eq!(report["host_to_device_upload_bytes"], 737_280);
    assert_eq!(report["total_transfer_bytes"], 2_211_840);
}

#[test]
fn browser_turbo_packed_f16_preload_requires_all_objects_and_no_materialization_correctness() {
    let plan = validate_browser_packed_f16_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .unwrap();
    let before = PackedF16DenoiserCacheAudit::default();
    let after = PackedF16DenoiserCacheAudit {
        state: PackedF16DenoiserCacheState::Ready,
        packed_cache_ready: true,
        cached_stage_count: 46,
        cached_object_count: 106,
        cached_tensor_count: 912,
        retained_packed_bytes: 19_870_010_624,
        packed_read_bytes: 19_870_166_528,
        packed_upload_bytes: 19_870_010_624,
        materialization_packed_read_bytes: 0,
        materialized_stage_count: 0,
        object_unpack_count: 0,
        f32_write_bytes: 0,
        preload_attempt_count: 1,
        failure_count: 0,
    };
    validate_packed_f16_denoiser_preload(BooguVariant::Image01Turbo, plan, before, after).unwrap();
    let mut partial = after;
    partial.cached_tensor_count -= 1;
    assert!(
        validate_packed_f16_denoiser_preload(BooguVariant::Image01Turbo, plan, before, partial,)
            .is_err()
    );
    let mut widened_during_preload = after;
    widened_during_preload.materialized_stage_count = 1;
    assert!(
        validate_packed_f16_denoiser_preload(
            BooguVariant::Image01Turbo,
            plan,
            before,
            widened_during_preload,
        )
        .is_err()
    );
}

#[test]
fn browser_turbo_packed_f16_repeat_rehydrates_after_request_scoped_eviction_correctness() {
    let plan = validate_browser_packed_f16_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16QwenVisionF32,
    )
    .unwrap();
    let empty_after_first_handoff = PackedF16DenoiserCacheAudit {
        state: PackedF16DenoiserCacheState::Empty,
        packed_cache_ready: false,
        cached_stage_count: 0,
        cached_object_count: 0,
        cached_tensor_count: 0,
        retained_packed_bytes: 0,
        packed_read_bytes: 19_870_166_528,
        packed_upload_bytes: 19_870_010_624,
        materialization_packed_read_bytes: 79_480_042_496,
        materialized_stage_count: 184,
        object_unpack_count: 424,
        f32_write_bytes: 158_960_084_992,
        preload_attempt_count: 1,
        failure_count: 0,
    };
    let before_second_dmd = PackedF16DenoiserCacheAudit {
        state: PackedF16DenoiserCacheState::Ready,
        packed_cache_ready: true,
        cached_stage_count: 46,
        cached_object_count: 106,
        cached_tensor_count: 912,
        retained_packed_bytes: 19_870_010_624,
        packed_read_bytes: 39_740_333_056,
        packed_upload_bytes: 39_740_021_248,
        materialization_packed_read_bytes: 79_480_042_496,
        materialized_stage_count: 184,
        object_unpack_count: 424,
        f32_write_bytes: 158_960_084_992,
        preload_attempt_count: 2,
        failure_count: 0,
    };
    validate_packed_f16_denoiser_preload(
        BooguVariant::Image01Turbo,
        plan,
        empty_after_first_handoff,
        before_second_dmd,
    )
    .unwrap();
    let after_second_dmd = PackedF16DenoiserCacheAudit {
        materialization_packed_read_bytes: 158_960_084_992,
        materialized_stage_count: 368,
        object_unpack_count: 848,
        f32_write_bytes: 317_920_169_984,
        ..before_second_dmd
    };
    let report = packed_f16_lifecycle_report(
        BooguVariant::Image01Turbo,
        before_second_dmd,
        after_second_dmd,
        BrowserArtifactTrafficReport::default(),
        false,
    )
    .unwrap();
    let report =
        validate_packed_f16_denoiser_lifecycle(BooguVariant::Image01Turbo, plan, 4, report)
            .unwrap();
    assert!(report.matches_plan);
    assert_eq!(report.preload_attempt_count, 2);
    assert_eq!(report.authenticated_artifact_bytes, 39_740_333_056);
    assert_eq!(report.packed_upload_bytes, 39_740_021_248);
    assert_eq!(report.stage_materializations, 184);
    assert_eq!(report.object_unpacks, 424);
    assert_eq!(report.packed_read_bytes, 79_480_042_496);
    assert_eq!(report.f32_write_bytes, 158_960_084_992);
    assert_eq!(
        report.dmd_artifact_traffic,
        BrowserArtifactTrafficReport::default()
    );

    let empty_after_second_handoff = PackedF16DenoiserCacheAudit {
        state: PackedF16DenoiserCacheState::Empty,
        packed_cache_ready: false,
        cached_stage_count: 0,
        cached_object_count: 0,
        cached_tensor_count: 0,
        retained_packed_bytes: 0,
        ..after_second_dmd
    };
    assert_eq!(
        packed_f16_cache_evidence(empty_after_second_handoff),
        BrowserPackedF16CacheEvidence {
            state: "empty".into(),
            cache_ready: false,
            cached_stages: 0,
            cached_objects: 0,
            cached_tensors: 0,
            cached_bytes: 0,
        }
    );
}

fn mixed_f16_settings() -> crate::BooguAdapterSettings {
    crate::BooguAdapterSettings::mixed_f16(ArtifactSource::Remote {
        base_url: RemoteBaseUrl::new("https://models.example/production").unwrap(),
    })
}

fn q4_settings() -> crate::BooguAdapterSettings {
    crate::BooguAdapterSettings::production(
        BooguVariant::Image01Turbo,
        ArtifactSource::Remote {
            base_url: RemoteBaseUrl::new("https://models.example/q4").unwrap(),
        },
    )
}

#[test]
fn browser_exact_f32_qualification_is_truthfully_labeled_and_not_eager_correctness() {
    let policy = BrowserExecutionPolicies::exact_1k5_parity(&mixed_f16_settings());
    assert_eq!(
        policy.residency,
        BrowserBooguResidencyPolicy::QualificationPerRequestF32DenoiserRetained
    );
    assert_eq!(policy.qwen_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(policy.vae_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(policy.denoiser_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(
        policy.denoiser_runtime_quantization,
        BooguDenoiserRuntimeQuantizationPolicy::Disabled
    );
    assert!(!policy.retain_qwen_stages);
    assert!(!policy.retain_vae_stages);
    assert!(policy.retain_denoiser_stages);
    assert!(!policy.eager_preload);
    assert!(!policy.defer_retained_qwen_synchronization);
    assert!(!policy.defer_retained_denoiser_synchronization);
    assert!(!policy.release_unused_qwen_memory_after_stage);
    assert!(!policy.packed_qwen_instruction_handoff);
    assert_eq!(
        policy.qwen_text_block_load_synchronization,
        Qwen3VlTextBlockLoadSynchronizationPolicy::PostForwardOnly
    );
    assert!(policy.packed_allocator_policy_is_exact());
    let mut unexpected_pre_forward_barrier = policy;
    unexpected_pre_forward_barrier.qwen_text_block_load_synchronization =
        Qwen3VlTextBlockLoadSynchronizationPolicy::PreForwardAndPostForward;
    assert!(!unexpected_pre_forward_barrier.packed_allocator_policy_is_exact());
    assert_eq!(
        policy.weight_traffic_contract(),
        "qualification-per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4"
    );
}

#[test]
fn browser_low_vram_policy_streams_qwen_vae_and_retains_only_runtime_q8_denoiser_correctness() {
    let settings = mixed_f16_settings();
    let policy = BrowserExecutionPolicies::low_vram_runtime_q8_denoiser(
        BooguVariant::Image01EditTurbo1k5,
        &settings,
    )
    .unwrap();
    assert_eq!(
        policy.residency,
        BrowserBooguResidencyPolicy::LowVramRuntimeQ8Denoiser
    );
    assert_eq!(policy.qwen_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(policy.qwen_quantized, BooguQuantizedLoadPolicy::Preserve);
    assert_eq!(policy.vae_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(policy.denoiser_float, BooguFloatLoadPolicy::AdaptToF32);
    assert_eq!(
        policy.denoiser_quantized,
        BooguQuantizedLoadPolicy::Preserve
    );
    assert_eq!(
        policy.denoiser_runtime_quantization,
        BooguDenoiserRuntimeQuantizationPolicy::Q8sBlock32F32
    );
    assert_eq!(
        denoiser_quantized_policy_name(
            policy.denoiser_quantized,
            policy.denoiser_runtime_quantization,
        ),
        "runtime-quantize-q8s-block32-f32"
    );
    assert_eq!(
        policy.denoiser_retaining_wrapper_adapter,
        BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul
    );
    assert!(!policy.retain_qwen_stages);
    assert!(!policy.retain_vae_stages);
    assert!(policy.retain_denoiser_stages);
    assert!(!policy.eager_preload);
    assert!(!policy.defer_retained_qwen_synchronization);
    assert!(!policy.defer_retained_denoiser_synchronization);
    assert!(!policy.release_unused_qwen_memory_after_stage);
    assert!(!policy.packed_qwen_instruction_handoff);
    assert!(policy.packed_allocator_policy_is_exact());
    assert_eq!(
        policy.weight_traffic_contract(),
        "per-request/qwen+vae+denoiser-first-dmd-step/denoiser-cache-hits-steps-2-through-4"
    );
    let ordinary = policy.for_ordinary_browser_factory(true);
    assert!(ordinary.require_persistent_range_cache);
    assert!(ordinary.request_scoped_surface_acquire_suspended);
    assert_eq!(
        ordinary.weight_traffic_contract(),
        "persistent-transport-part-cache/qwen+vae+denoiser-first-request/zero-repeat-network-required/retained-q8-direct-matmul-denoiser-cache-hits-dmd-steps-2-through-4"
    );

    let exact = BrowserExecutionPolicies::exact_1k5_low_vram_parity(&settings).unwrap();
    assert!(!exact.request_scoped_surface_acquire_suspended);
    assert!(
        !exact
            .provenance_backend()
            .contains(BROWSER_SURFACE_INFERENCE_PROVENANCE_SUFFIX)
    );
    assert_eq!(exact.residency, policy.residency);
    assert_eq!(exact.denoiser_quantized, policy.denoiser_quantized);
    assert_eq!(
        exact.denoiser_runtime_quantization,
        policy.denoiser_runtime_quantization
    );
    assert_eq!(
        denoiser_quantized_policy_name(
            exact.denoiser_quantized,
            exact.denoiser_runtime_quantization,
        ),
        "runtime-quantize-q8s-block32-f32"
    );
    assert_eq!(exact.retain_denoiser_stages, policy.retain_denoiser_stages);
    assert!(!exact.defer_retained_qwen_synchronization);
    assert!(!exact.defer_retained_denoiser_synchronization);
    assert!(!exact.require_persistent_range_cache);

    let mut wrong_profile = settings;
    wrong_profile.storage_profile = BooguStorageProfile::F16;
    assert!(
        BrowserExecutionPolicies::low_vram_runtime_q8_denoiser(
            BooguVariant::Image01Turbo,
            &mixed_f16_settings(),
        )
        .unwrap_err()
        .contains("not numerically qualified")
    );
    let error = BrowserExecutionPolicies::low_vram_runtime_q8_denoiser(
        BooguVariant::Image01EditTurbo,
        &wrong_profile,
    )
    .unwrap_err();
    assert!(error.contains("profile=production"), "{error}");
}

#[test]
fn browser_artifact_traffic_report_preserves_cache_and_network_counters_correctness() {
    let snapshot = BrowserArtifactTrafficSnapshot {
        object_reads: 1,
        object_read_bytes: 2,
        range_fetch_requests: 3,
        range_response_bytes: 4,
        verified_objects: 5,
        cache_lookup_requests: 6,
        cache_hits: 7,
        cache_misses: 8,
        cache_read_bytes: 9,
        network_fetch_requests: 10,
        network_response_bytes: 11,
        cache_write_requests: 12,
        cache_write_bytes: 13,
        cache_eviction_requests: 14,
        cache_evicted_entries: 15,
        cache_invalid_entries: 16,
        integrity_refetches: 17,
    };
    let report = BrowserArtifactTrafficReport::from(snapshot);
    assert_eq!(report.object_reads, 1);
    assert_eq!(report.object_read_bytes, 2);
    assert_eq!(report.range_reads, 3);
    assert_eq!(report.range_read_bytes, 4);
    assert_eq!(report.verified_objects, 5);
    assert_eq!(report.cache_lookups, 6);
    assert_eq!(report.cache_hits, 7);
    assert_eq!(report.cache_misses, 8);
    assert_eq!(report.cache_read_bytes, 9);
    assert_eq!(report.network_requests, 10);
    assert_eq!(report.network_response_bytes, 11);
    assert_eq!(report.cache_writes, 12);
    assert_eq!(report.cache_write_bytes, 13);
    assert_eq!(report.cache_evictions, 14);
    assert_eq!(report.cache_evicted_entries, 15);
    assert_eq!(report.cache_invalid_entries, 16);
    assert_eq!(report.integrity_refetches, 17);
}

#[test]
fn browser_low_vram_streamed_qwen_vae_lifecycle_is_fail_closed_correctness() {
    let variant = BooguVariant::Image01EditTurbo1k5;
    validate_low_vram_streamed_stage_lifecycle(variant, 0, false, 0).unwrap();
    for result in [
        validate_low_vram_streamed_stage_lifecycle(variant, 1, false, 0),
        validate_low_vram_streamed_stage_lifecycle(variant, 0, true, 0),
        validate_low_vram_streamed_stage_lifecycle(variant, 0, false, 1),
    ] {
        assert!(result.is_err());
    }
}

#[test]
fn browser_q8_resource_math_covers_current_edit_variants_correctness() {
    let qwen_plan = released_qwen_streaming_plan();
    let inventory = BooguArtifactInventory::new(
        &released_qwen_config(),
        &BooguConfig::default(),
        &released_flux_vae_config(),
    )
    .unwrap();
    assert_eq!(
        max_streamed_qwen_stage_f32_bytes(BooguVariant::Image01Turbo, &qwen_plan).unwrap(),
        771_785_728
    );
    for variant in [
        BooguVariant::Image01EditTurbo,
        BooguVariant::Image01EditTurbo1k5,
    ] {
        let plan = validate_browser_low_vram_resource_plan(
            variant,
            BooguStorageProfile::F16QwenVisionF32,
            &qwen_plan,
            &inventory,
            BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul,
        )
        .unwrap();
        assert_eq!(plan.audited_retained_q8_denoiser_bytes, 12_590_785_792);
        assert_eq!(plan.audited_max_streamed_qwen_stage_f32_bytes, 771_785_728);
        assert_eq!(plan.audited_loaded_vae_module_f32_bytes, 335_278_732);
        assert_eq!(plan.audited_max_dense_denoiser_stage_f32_bytes, 0);
        assert_eq!(plan.audited_max_phase_local_f32_stage_bytes, 771_785_728);
        assert_eq!(plan.runtime_quantization_workspace_bytes, 2_434_252_800);
        assert_eq!(plan.activation_reserve_bytes, 14_605_516_800);
        assert_eq!(plan.conservative_planned_device_bytes, 30_402_341_120);
        assert_eq!(plan.expected_q8s_block32_f32_tensor_count, 377);
        assert_eq!(plan.expected_f32_tensor_count, 565);
        assert!(plan.conservative_planned_device_bytes < plan.strict_device_cap_bytes);
        let event = serde_json::to_value(browser_low_vram_resource_plan_event(
            plan,
            BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul,
        ))
        .unwrap();
        assert_eq!(
            event["denoiser_quantized_load_policy"],
            "runtime-quantize-q8s-block32-f32"
        );
        assert_eq!(
            event["denoiser_quantized_linear_execution_policy"],
            "direct-quantized-matmul"
        );
        assert_eq!(
            serde_json::to_value(plan).unwrap()["strict_device_cap_bytes"],
            BROWSER_LOW_VRAM_STRICT_DEVICE_CAP_BYTES
        );
    }

    let error = validate_browser_low_vram_resource_plan(
        BooguVariant::Image01Turbo,
        BooguStorageProfile::F16,
        &qwen_plan,
        &inventory,
        BooguQuantizedLinearExecutionPolicy::DenseF32PerSemanticStage,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("unchanged canonical profile=production"),
        "{error}"
    );
}

#[test]
fn browser_turbo_packed_f16_resident_weight_plan_is_exact_correctness() {
    let inventory = BooguArtifactInventory::new(
        &released_qwen_config(),
        &BooguConfig::default(),
        &released_flux_vae_config(),
    )
    .unwrap();
    let footprint = inventory
        .packed_f16_resident_footprint(
            BooguVariant::Image01Turbo,
            BooguStorageProfile::F16QwenVisionF32,
        )
        .unwrap();

    assert_eq!(footprint.packed_f16_tensor_count, 778);
    assert_eq!(footprint.f32_tensor_count, 670);
    assert_eq!(footprint.packed_f16_payload_bytes, 35_101_539_904);
    assert_eq!(footprint.f32_payload_bytes, 8_716_300);
    assert_eq!(footprint.total_payload_bytes, 35_110_256_204);
    assert_eq!(
        footprint.total_payload_bytes,
        footprint.packed_f16_payload_bytes + footprint.f32_payload_bytes
    );
    assert_eq!(
        footprint.total_payload_bytes
            + crate::boogu::BOOGU_BROWSER_MAX_APPLIED_BUFFER_BYTES
                * BROWSER_RESIDENT_MAX_SIMULTANEOUS_ACTIVATION_BUFFERS,
        43_408_096_844
    );
}

#[test]
fn browser_low_vram_resource_plan_fails_closed_at_cap_correctness() {
    let qwen_plan = released_qwen_streaming_plan();
    let inventory = BooguArtifactInventory::new(
        &released_qwen_config(),
        &BooguConfig::default(),
        &released_flux_vae_config(),
    )
    .unwrap();
    let accepted = validate_browser_low_vram_resource_plan(
        BooguVariant::Image01EditTurbo1k5,
        BooguStorageProfile::F16QwenVisionF32,
        &qwen_plan,
        &inventory,
        BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul,
    )
    .unwrap();
    let error = validate_browser_low_vram_resource_plan_with_cap(
        BooguVariant::Image01EditTurbo1k5,
        BooguStorageProfile::F16QwenVisionF32,
        &qwen_plan,
        &inventory,
        BooguQuantizedLinearExecutionPolicy::DirectQuantizedMatmul,
        accepted.conservative_planned_device_bytes,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("not strictly below"), "{error}");
}

#[test]
fn browser_low_vram_harness_peaks_interval_totals_not_individual_rows_correctness() {
    let harness = include_str!("../tests/wasm_browser_1k5_parity.mjs");
    assert!(harness.contains(
            "const totalFramebufferMib = rows.reduce(\n    (total, row) => total + (row.framebuffer_mib ?? 0),"
        ));
    assert!(harness.contains(
            "evidence.max_framebuffer_mib = Math.max(evidence.max_framebuffer_mib, totalFramebufferMib);"
        ));
    assert!(harness.contains("total_framebuffer_mib: totalFramebufferMib"));
    assert!(!harness.contains(
        "evidence.max_framebuffer_mib = Math.max(evidence.max_framebuffer_mib, row.framebuffer_mib"
    ));
}

#[test]
fn browser_edit_q8_denoiser_lifecycle_is_exactly_four_steps_correctness() {
    let variant = BooguVariant::Image01EditTurbo1k5;
    // Request-scoped Edit/1.5K residency clears all stages before VAE decode.
    validate_low_vram_denoiser_lifecycle(
        variant,
        4,
        BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
        BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
        false,
        0,
        0,
    )
    .unwrap();
    for error in [
        validate_low_vram_denoiser_lifecycle(
            variant,
            3,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            false,
            0,
            0,
        ),
        validate_low_vram_denoiser_lifecycle(
            variant,
            4,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT - 1,
            false,
            0,
            0,
        ),
        validate_low_vram_denoiser_lifecycle(
            variant,
            4,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            true,
            0,
            0,
        ),
        validate_low_vram_denoiser_lifecycle(
            variant,
            4,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            BROWSER_1K5_DENOISER_RETAINED_STAGE_COUNT,
            false,
            0,
            1,
        ),
    ] {
        assert!(error.is_err());
    }
}

#[test]
fn browser_webgpu_vae_f32_oracle_envelope_is_complete_and_scoped_correctness() {
    let envelope = BROWSER_WEBGPU_VAE_F32_ORACLE_ENVELOPE;
    assert_eq!(
        envelope.artifact_content_digest,
        BROWSER_WEBGPU_VAE_F32_ORACLE_SOURCE_CONTENT_DIGEST
    );
    assert_eq!(envelope.weight_storage_dtype, "f16");
    assert_eq!(envelope.weight_load_policy, "adapt-to-f32");
    assert_eq!(envelope.execution_dtype, "f32");
    assert_eq!(envelope.portability, "no-cross-adapter-portability-claim");
    assert_eq!(envelope.moments.maximum_abs, 0.016);
    assert_eq!(envelope.moments.maximum_rmse, 0.000_75);
    assert_eq!(envelope.mean.maximum_abs, 0.013);
    assert_eq!(envelope.logvar.maximum_abs, 0.016);
    assert_eq!(envelope.std.maximum_abs, 0.000_1);
    assert_eq!(envelope.raw_latent.maximum_abs, 0.013);
    assert_eq!(envelope.scaled_latent.maximum_abs, 0.005);
    assert_eq!(envelope.scaled_latent.maximum_rmse, 0.000_2);
    assert_eq!(
        envelope.component_maximum("vae.reference_f32_scaled_latent"),
        Some(envelope.scaled_latent.maximum_abs)
    );
    assert_eq!(
        envelope.component_maximum("vae.reference_f32_unknown"),
        None
    );

    let serialized = serde_json::to_value(envelope).unwrap();
    assert_eq!(
        serialized["calibrated_device"],
        serde_json::Value::String("0x2bb1".into())
    );
    assert_eq!(serialized["moments"]["maximum_abs"], 0.016);
    assert_eq!(serialized["scaled_latent"]["maximum_abs"], 0.005);
}

#[test]
fn canonical_digest_requirement_tracks_exact_origin_correctness() {
    let variant = BooguVariant::Image01Turbo;
    let profile = BooguStorageProfile::F16QwenVisionF32;
    let canonical = burn_image::RemoteBaseUrl::new(format!(
        "{}/boogu-image-0.1-turbo",
        crate::boogu::BOOGU_CDN_ROOT
    ))
    .unwrap();
    let source = burn_image::RemoteBaseUrl::new(format!(
        "{}/boogu-image-0.1-turbo-f16-qwen-vision-f32",
        crate::boogu::BOOGU_CDN_ROOT
    ))
    .unwrap();
    let custom = burn_image::RemoteBaseUrl::new(
        "https://models.example/boogu-image-0.1-turbo-f16-qwen-vision-f32",
    )
    .unwrap();
    assert!(browser_source_requires_canonical_digest(
        variant, profile, &canonical
    ));
    assert!(!browser_source_requires_canonical_digest(
        variant, profile, &source
    ));
    assert!(!browser_source_requires_canonical_digest(
        variant, profile, &custom
    ));
    assert!(!browser_source_requires_canonical_digest(
        variant,
        BooguStorageProfile::F16,
        &canonical
    ));
}

#[test]
fn browser_composition_requires_exact_component_roles_correctness() {
    let qwen = tiny_manifest("shared-qwen", ARTIFACT_MANIFEST_SCHEMA_V1);
    let vae = tiny_manifest("shared-vae", ARTIFACT_MANIFEST_SCHEMA_V1);
    let mut parent = tiny_manifest("pipeline", ARTIFACT_MANIFEST_SCHEMA_V2);
    parent.dependencies = vec![dependency("qwen", &qwen), dependency("vae", &vae)];
    parent
        .metadata
        .insert("component_dependency_count".into(), "2".into());
    parent.content_digest = None;
    parent.seal().unwrap();
    let variant = BooguVariant::Image01Turbo;
    assert_eq!(
        browser_dependency(&parent, "qwen", variant).unwrap().bundle,
        qwen.bundle
    );
    assert_eq!(
        browser_dependency(&parent, "vae", variant).unwrap().bundle,
        vae.bundle
    );

    parent.dependencies.pop();
    let error = browser_dependency(&parent, "vae", variant)
        .unwrap_err()
        .to_string();
    assert!(error.contains("omits required vae dependency"), "{error}");
}

#[test]
fn browser_weight_ledger_qualifies_component_bundle_paths_correctness() {
    let manifest = tiny_manifest("shared-qwen", ARTIFACT_MANIFEST_SCHEMA_V1);
    let unqualified = manifest_weight_artifacts(&manifest, false);
    let qualified = manifest_weight_artifacts(&manifest, true);
    // Tiny fixture uses metadata; make sure both routes remain empty rather
    // than accidentally counting compact bootstrap files as model weights.
    assert!(unqualified.is_empty());
    assert!(qualified.is_empty());

    let mut weights = manifest;
    weights.files[0].role = ArtifactFileRole::Weights;
    assert!(manifest_weight_artifacts(&weights, false).contains_key("objects/tiny.bpk"));
    assert!(manifest_weight_artifacts(&weights, true).contains_key("shared-qwen/objects/tiny.bpk"));
}

#[test]
fn canonical_turbo_active_transfer_denominator_is_exact_correctness() {
    let exact = burn_image::ArtifactTransferProgress {
        phase: "Model setup".into(),
        component: None,
        logical_objects_completed: BROWSER_TURBO_ACTIVE_LOGICAL_OBJECTS,
        logical_objects_total: BROWSER_TURBO_ACTIVE_LOGICAL_OBJECTS,
        physical_parts_completed: BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS,
        physical_parts_total: BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS,
        bounded_ranges_completed: u64::from(BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS),
        bounded_ranges_total: u64::from(BROWSER_TURBO_ACTIVE_UNIQUE_TRANSPORT_PARTS),
        loaded_bytes: BROWSER_TURBO_ACTIVE_TRANSPORT_BYTES,
        total_bytes: BROWSER_TURBO_ACTIVE_TRANSPORT_BYTES,
        bytes_per_second: None,
        eta_seconds: None,
        request_activity: None,
    };
    validate_browser_turbo_active_transfer_plan(&exact).unwrap();

    for drift in [
        (185, 1_751, 35_106_151_424),
        (186, 1_752, 35_106_151_424),
        (186, 1_751, 35_106_151_425),
    ] {
        let mut invalid = exact.clone();
        invalid.logical_objects_total = drift.0;
        invalid.physical_parts_total = drift.1;
        invalid.total_bytes = drift.2;
        assert!(validate_browser_turbo_active_transfer_plan(&invalid).is_err());
    }
}

#[test]
fn custom_browser_source_rejects_arbitrary_bundle_identity_correctness() {
    let variant = BooguVariant::Image01Turbo;
    let profile = BooguStorageProfile::F16QwenVisionF32;

    validate_browser_manifest_bundle_identity(variant, profile, "boogu-image-0.1-turbo").unwrap();
    validate_browser_manifest_bundle_identity(
        variant,
        profile,
        "boogu-image-0.1-turbo-f16-qwen-vision-f32",
    )
    .unwrap();
    let error = validate_browser_manifest_bundle_identity(
        variant,
        profile,
        "boogu-image-0.1-turbo-arbitrary",
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("incompatible with the selected release/profile"),
        "{error}"
    );
    assert!(
        error.contains(
            "expected boogu-image-0.1-turbo or explicit conversion source boogu-image-0.1-turbo-f16-qwen-vision-f32"
        ),
        "{error}"
    );
}
