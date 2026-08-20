//! Prepare the modular Boogu production CDN release from exact conversion-source bundles.
//!
//! Each release keeps the three public Boogu bundle ids while moving byte-identical Qwen3-VL and
//! FLUX VAE contracts into sibling components. Packed-Q4S releases share one Q4S Qwen component
//! and one F16 VAE component across all three denoiser-only parents. Every output receives a
//! sealed part-only transport layout; this command never rewrites a source bundle or existing
//! upload tree.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
};

use burn_boogu::{
    BooguVariant,
    artifacts::{
        BooguReleaseVerification, BooguStorageProfile, release_source_artifact_digest,
        source_artifact_bundle_id, verify_modular_release_artifact_directories,
        verify_release_artifact_directory,
    },
    boogu_model_descriptor,
};
use burn_flux_vae::{
    FLUX_VAE_COMPONENT_MODEL_ID, FLUX_VAE_COMPONENT_MODEL_REVISION, FLUX_VAE_COMPONENT_PROFILE,
    FLUX_VAE_COMPONENT_ROLE, FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID,
};
use burn_image::{
    ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_MANIFEST_SCHEMA_V2, ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES,
    ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY, ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY,
    ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY, ARTIFACT_TRANSPORT_LAYOUT_PATH,
    ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY, ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY,
    ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION, ARTIFACT_TRANSPORT_MAX_PART_BYTES,
    ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY, ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY,
    ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactBundleId, ArtifactComponent, ArtifactComponentId,
    ArtifactDependency, ArtifactFile, ArtifactFileRole, ArtifactManifest, ArtifactPath,
    ArtifactProfileId, ArtifactTransportLayout, ArtifactTransportObject, ArtifactTransportPart,
    ModelId, NumericFormat, Sha256Digest,
};
use burn_qwen3_vl::{
    QWEN_BASE_CONDITIONING_PROFILE, QWEN_COMPONENT_BUNDLE_ID, QWEN_COMPONENT_MODEL_ID,
    QWEN_COMPONENT_MODEL_REVISION, QWEN_COMPONENT_ROLE, QWEN_Q4S_BASE_CONDITIONING_PROFILE,
    QWEN_Q4S_COMPONENT_BUNDLE_ID,
};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const CDN_ROOT: &str = "https://aberration.technology/model";
const CACHE_CONTROL: &str = "public,max-age=31536000,immutable";
const INVENTORY_PATH: &str = "metadata/tensor-inventory.json";
const SOURCE_FILES_PATH: &str = "metadata/source-files.json";
const VAE_CONFIG_PATH: &str = "metadata/source/vae/config.json";
const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
// Every object published directly to the CDN, not only transport parts, must fit the same
// browser-friendly physical ceiling. Logical weight objects retain their separate semantic cap
// because the release tree contains only their bounded transport parts.
const MAX_METADATA_BYTES: u64 = ARTIFACT_TRANSPORT_MAX_PART_BYTES;

#[derive(Debug, Parser)]
#[command(about = "Build the dependency-pinned modular Boogu CDN release")]
struct Args {
    /// Directory containing the three mixed-F16 and three Q4S sealed semantic sources.
    #[arg(long, default_value = ".artifacts")]
    artifact_root: PathBuf,
    /// Fresh upload staging root. It must not already exist.
    #[arg(long, default_value = ".artifacts/cdn-upload-modular")]
    output_root: PathBuf,
    /// Copy payloads rather than requiring same-filesystem hardlinks.
    #[arg(long, default_value_t = false)]
    copy: bool,
    /// Stage only the dependency-first five-bundle Q4S release.
    #[arg(long, default_value_t = false)]
    q4_only: bool,
}

#[derive(Clone, Copy)]
struct BundleSpec {
    variant: BooguVariant,
    canonical_id: &'static str,
}

const BUNDLES: [BundleSpec; 3] = [
    BundleSpec {
        variant: BooguVariant::Image01Turbo,
        canonical_id: "boogu-image-0.1-turbo",
    },
    BundleSpec {
        variant: BooguVariant::Image01EditTurbo,
        canonical_id: "boogu-image-0.1-edit-turbo",
    },
    BundleSpec {
        variant: BooguVariant::Image01EditTurbo1k5,
        canonical_id: "boogu-image-0.1-edit-turbo-1k5",
    },
];

const Q4_BUNDLES: [BundleSpec; 3] = [
    BundleSpec {
        variant: BooguVariant::Image01Turbo,
        canonical_id: "boogu-image-0.1-turbo-q4s-block-up-to128-f32",
    },
    BundleSpec {
        variant: BooguVariant::Image01EditTurbo,
        canonical_id: "boogu-image-0.1-edit-turbo-q4s-block-up-to128-f32",
    },
    BundleSpec {
        variant: BooguVariant::Image01EditTurbo1k5,
        canonical_id: "boogu-image-0.1-edit-turbo-1k5-q4s-block-up-to128-f32",
    },
];

struct SourceBundle {
    spec: BundleSpec,
    id: String,
    directory: PathBuf,
    manifest: ArtifactManifest,
    verification: BooguReleaseVerification,
    inventory: Vec<Value>,
    source_files: Vec<Value>,
    normalized_vae_config: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Qwen,
    Denoiser,
    Vae,
}

impl Owner {
    const fn inventory_name(self) -> &'static str {
        match self {
            Self::Qwen => "qwen3-vl",
            Self::Denoiser => "boogu-denoiser",
            Self::Vae => "flux-vae",
        }
    }

    fn owns_component(self, component: &str) -> bool {
        match self {
            Self::Qwen => component.starts_with("qwen-"),
            Self::Denoiser => component.starts_with("boogu-"),
            Self::Vae => component.starts_with("flux-vae-"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BrowserBounds {
    range_chunk_bytes: u64,
    maximum_response_bytes: u64,
    maximum_manifest_bytes: u64,
    maximum_metadata_bytes: u64,
    maximum_semantic_object_bytes: u64,
    transport_part_target_bytes: u64,
    maximum_transport_part_bytes: u64,
}

#[derive(Debug, Serialize)]
struct BundlePlan {
    bundle_id: String,
    kind: &'static str,
    profile: String,
    model: String,
    model_revision: String,
    content_digest: String,
    cdn_base_url: String,
    manifest_url: String,
    local_directory: String,
    files: usize,
    weight_objects: usize,
    payload_bytes: u64,
    largest_payload_bytes: u64,
    transport_objects: usize,
    transport_parts: usize,
    transport_payload_bytes: u64,
    largest_transport_part_bytes: u64,
    transport_layout_bytes: u64,
    manifest_bytes: u64,
    dependencies: Vec<String>,
    browser_transport_fit: bool,
    browser_bounds: BrowserBounds,
}

#[derive(Debug, Serialize)]
struct UploadPhase {
    sequence: u8,
    name: &'static str,
    bundles: Vec<String>,
    include: &'static str,
    cache_control: &'static str,
    prerequisite: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct UploadPlan {
    schema_version: u32,
    release: &'static str,
    cdn_root: &'static str,
    generated_from: String,
    cache_control: &'static str,
    dependency_first: bool,
    manifest_last: bool,
    bundle_count: usize,
    bundles: Vec<BundlePlan>,
    upload_phases: Vec<UploadPhase>,
}

#[derive(Debug, Serialize)]
struct SharedContractProof {
    schema_version: u32,
    source_bundles: Vec<String>,
    qwen_declarations_identical: bool,
    qwen_upstream_sources_identical: bool,
    qwen_weight_objects: usize,
    qwen_weight_bytes: u64,
    vae_declarations_identical: bool,
    vae_upstream_source_identical: bool,
    vae_config_semantically_identical_after_provenance_normalization: bool,
    vae_weight_objects: usize,
    vae_weight_bytes: u64,
    denoiser_payloads_pairwise_disjoint: bool,
    reconstructed_source_closures_exact: bool,
    dependency_closures_verified: bool,
    component_contracts_verified: bool,
    bounded_burnpacks_verified: bool,
    bounded_transport_parts_verified: bool,
    transport_part_target_bytes: u64,
    maximum_transport_part_bytes: u64,
    duplicate_shared_bytes_removed: u64,
    component_revision_algorithm: &'static str,
    qwen_manifest_digest: String,
    q4_qwen_manifest_digest: Option<String>,
    vae_manifest_digest: String,
    pipeline_manifest_digests: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct PreparationSummary {
    output_root: String,
    upload_tree: String,
    upload_plan: String,
    upload_plan_sha256: String,
    equivalence_report: String,
    bundles: usize,
    logical_declared_payload_bytes: u64,
    physical_transport_parts: usize,
    physical_transport_payload_bytes: u64,
    largest_transport_part_bytes: u64,
    duplicate_shared_bytes_removed: u64,
}

#[derive(Clone, Copy, Debug)]
struct TransportStats {
    objects: usize,
    parts: usize,
    bytes: u64,
    largest_part_bytes: u64,
    layout_bytes: u64,
}

struct Cleanup {
    path: PathBuf,
    armed: bool,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if self.armed && self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Args::parse();
    args.artifact_root = fs::canonicalize(&args.artifact_root)?;
    if args.output_root.is_relative() {
        args.output_root = std::env::current_dir()?.join(&args.output_root);
    }
    let output_name = args
        .output_root
        .file_name()
        .ok_or("output root must end in a normal directory name")?
        .to_owned();
    let output_parent = args
        .output_root
        .parent()
        .ok_or("output root must have a parent directory")?;
    fs::create_dir_all(output_parent)?;
    args.output_root = fs::canonicalize(output_parent)?.join(output_name);
    prepare(&args)
}

fn prepare(args: &Args) -> Result<(), Box<dyn Error>> {
    require_real_directory(&args.artifact_root)?;
    if args.output_root.exists() {
        return Err(format!(
            "output root already exists; refusing to overwrite: {}",
            args.output_root.display()
        )
        .into());
    }

    let output_parent = args
        .output_root
        .parent()
        .ok_or("output root must have a parent directory")?;
    let output_stage = create_temporary_directory(output_parent, ".cdn-upload-modular.prepare")?;
    let cleanup = Cleanup {
        path: output_stage.clone(),
        armed: true,
    };
    let cdn_tree = output_stage.join("aberration.technology/model");
    let reports_root = output_stage.join("verification-reports");
    fs::create_dir_all(&cdn_tree)?;
    fs::create_dir_all(&reports_root)?;

    let q4_sources = read_and_verify_q4_bundles(args)?;
    for source in &q4_sources {
        if source.verification.variant != source.spec.variant
            || source.verification.profile != BooguStorageProfile::Q4sBlockUpTo128F32
            || source.verification.verified_files != source.manifest.files.len()
        {
            return Err(format!(
                "{} Q4S semantic verification did not cover the exact source release",
                source.id
            )
            .into());
        }
    }
    if args.q4_only {
        return prepare_q4_only(
            args,
            output_stage,
            cdn_tree,
            reports_root,
            cleanup,
            &q4_sources,
        );
    }

    let sources = read_and_verify_source_bundles(args)?;
    for bundle in &sources {
        if bundle.verification.variant != bundle.spec.variant
            || bundle.verification.profile != BooguStorageProfile::F16QwenVisionF32
            || bundle.verification.verified_files != bundle.manifest.files.len()
        {
            return Err(format!(
                "{} semantic verification did not cover the exact expected release",
                bundle.id
            )
            .into());
        }
    }
    prove_shared_inputs(&sources)?;
    prove_denoisers_disjoint(&sources)?;
    prove_shared_inputs(&q4_sources)?;
    prove_denoisers_disjoint(&q4_sources)?;

    let reference = &sources[0];
    let qwen_manifest = stage_component_bundle(
        &cdn_tree,
        reference,
        ComponentSpec {
            owner: Owner::Qwen,
            bundle: QWEN_COMPONENT_BUNDLE_ID,
            profile: QWEN_BASE_CONDITIONING_PROFILE,
            model: QWEN_COMPONENT_MODEL_ID,
            model_revision: QWEN_COMPONENT_MODEL_REVISION,
            numeric_format: NumericFormat::Other(QWEN_BASE_CONDITIONING_PROFILE.to_owned()),
            bind_source_bundle: true,
        },
        args.copy,
    )?;
    let vae_manifest = stage_component_bundle(
        &cdn_tree,
        reference,
        ComponentSpec {
            owner: Owner::Vae,
            bundle: FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID,
            profile: FLUX_VAE_COMPONENT_PROFILE,
            model: FLUX_VAE_COMPONENT_MODEL_ID,
            model_revision: FLUX_VAE_COMPONENT_MODEL_REVISION,
            numeric_format: NumericFormat::F16,
            bind_source_bundle: false,
        },
        args.copy,
    )?;

    let qwen_dependency = dependency_from_manifest(QWEN_COMPONENT_ROLE, &qwen_manifest)?;
    let vae_dependency = dependency_from_manifest(FLUX_VAE_COMPONENT_ROLE, &vae_manifest)?;
    let mut pipeline_manifests = Vec::with_capacity(sources.len());
    for source in &sources {
        pipeline_manifests.push(stage_pipeline_bundle(
            &cdn_tree,
            source,
            &[qwen_dependency.clone(), vae_dependency.clone()],
            args.copy,
        )?);
    }
    let q4_reference = &q4_sources[0];
    prove_cross_profile_vae(reference, q4_reference)?;
    let q4_qwen_manifest = stage_component_bundle(
        &cdn_tree,
        q4_reference,
        ComponentSpec {
            owner: Owner::Qwen,
            bundle: QWEN_Q4S_COMPONENT_BUNDLE_ID,
            profile: QWEN_Q4S_BASE_CONDITIONING_PROFILE,
            model: QWEN_COMPONENT_MODEL_ID,
            model_revision: QWEN_COMPONENT_MODEL_REVISION,
            numeric_format: NumericFormat::Other(QWEN_Q4S_BASE_CONDITIONING_PROFILE.to_owned()),
            bind_source_bundle: true,
        },
        args.copy,
    )?;
    let q4_qwen_dependency = dependency_from_manifest(QWEN_COMPONENT_ROLE, &q4_qwen_manifest)?;
    let q4_manifests = q4_sources
        .iter()
        .map(|source| {
            stage_pipeline_bundle(
                &cdn_tree,
                source,
                &[q4_qwen_dependency.clone(), vae_dependency.clone()],
                args.copy,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let candidate_manifest_digests = std::iter::once(&qwen_manifest)
        .chain(std::iter::once(&q4_qwen_manifest))
        .chain(std::iter::once(&vae_manifest))
        .chain(pipeline_manifests.iter())
        .chain(q4_manifests.iter())
        .map(|manifest| (manifest.bundle.to_string(), sealed_digest(manifest)))
        .collect::<BTreeMap<_, _>>();
    write_json_new(
        &reports_root.join("candidate-manifest-digests.json"),
        &candidate_manifest_digests,
    )?;
    eprintln!(
        "candidate transport-sealed manifest digests:\n{}",
        serde_json::to_string_pretty(&candidate_manifest_digests)?
    );

    let manifests = std::iter::once(&qwen_manifest)
        .chain(std::iter::once(&q4_qwen_manifest))
        .chain(std::iter::once(&vae_manifest))
        .chain(pipeline_manifests.iter())
        .chain(q4_manifests.iter())
        .collect::<Vec<_>>();
    let resolved = manifests
        .iter()
        .map(|manifest| (manifest.bundle.clone(), *manifest))
        .collect::<BTreeMap<_, _>>();
    for pipeline in pipeline_manifests.iter().chain(q4_manifests.iter()) {
        pipeline.validate_dependency_closure(|bundle| resolved.get(bundle).copied())?;
    }
    for (source, pipeline) in sources.iter().zip(&pipeline_manifests) {
        prove_reconstructed_closure(source, pipeline, &qwen_manifest, &vae_manifest, &cdn_tree)?;
        let modular = verify_modular_release_artifact_directories(
            cdn_tree.join(pipeline.bundle.as_str()),
            cdn_tree.join(qwen_manifest.bundle.as_str()),
            cdn_tree.join(vae_manifest.bundle.as_str()),
        )?;
        if !modular.dependency_closure_verified
            || !modular.component_contracts_verified
            || !modular.reconstructed_inventory_verified
            || modular.verified_tensors != 1_940
            || modular.verified_weight_objects != 223
        {
            return Err(format!(
                "{} modular semantic verification is incomplete",
                pipeline.bundle
            )
            .into());
        }
        write_json_new(
            &reports_root.join(format!("{}-semantic.json", pipeline.bundle)),
            &modular,
        )?;
    }

    for (source, manifest) in q4_sources.iter().zip(&q4_manifests) {
        prove_reconstructed_closure(
            source,
            manifest,
            &q4_qwen_manifest,
            &vae_manifest,
            &cdn_tree,
        )?;
        let verification = verify_modular_release_artifact_directories(
            cdn_tree.join(manifest.bundle.as_str()),
            cdn_tree.join(q4_qwen_manifest.bundle.as_str()),
            cdn_tree.join(vae_manifest.bundle.as_str()),
        )?;
        require_matching_q4_modular_verification(&source.verification, &verification)?;
        write_json_new(
            &reports_root.join(format!("{}-semantic.json", manifest.bundle)),
            &verification,
        )?;
    }

    if fs::read_dir(&cdn_tree)?.count() != 9 {
        return Err(
            "CDN tree must contain exactly three shared components, three mixed-F16 pipelines, and three Q4S pipelines"
                .into(),
        );
    }

    let qwen_bytes = weight_bytes(&qwen_manifest)?;
    let q4_qwen_bytes = weight_bytes(&q4_qwen_manifest)?;
    let vae_bytes = weight_bytes(&vae_manifest)?;
    let duplicate_shared_bytes_removed = qwen_bytes
        .checked_add(q4_qwen_bytes)
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_add(vae_bytes.checked_mul(5)?))
        .ok_or("shared byte count overflow")?;
    let proof = SharedContractProof {
        schema_version: 2,
        source_bundles: sources
            .iter()
            .map(|bundle| bundle.id.clone())
            .chain(q4_sources.iter().map(|bundle| bundle.id.clone()))
            .collect(),
        qwen_declarations_identical: true,
        qwen_upstream_sources_identical: true,
        qwen_weight_objects: weight_count(&qwen_manifest),
        qwen_weight_bytes: qwen_bytes,
        vae_declarations_identical: true,
        vae_upstream_source_identical: true,
        vae_config_semantically_identical_after_provenance_normalization: true,
        vae_weight_objects: weight_count(&vae_manifest),
        vae_weight_bytes: vae_bytes,
        denoiser_payloads_pairwise_disjoint: true,
        reconstructed_source_closures_exact: true,
        dependency_closures_verified: true,
        component_contracts_verified: true,
        bounded_burnpacks_verified: true,
        bounded_transport_parts_verified: true,
        transport_part_target_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
        maximum_transport_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        duplicate_shared_bytes_removed,
        component_revision_algorithm: "sha256(compact-json(owner-filtered-source-files-sorted-by-path) + LF)",
        qwen_manifest_digest: sealed_digest(&qwen_manifest),
        q4_qwen_manifest_digest: Some(sealed_digest(&q4_qwen_manifest)),
        vae_manifest_digest: sealed_digest(&vae_manifest),
        pipeline_manifest_digests: pipeline_manifests
            .iter()
            .chain(q4_manifests.iter())
            .map(|manifest| (manifest.bundle.to_string(), sealed_digest(manifest)))
            .collect(),
    };
    write_json_new(&reports_root.join("modular-equivalence.json"), &proof)?;

    let mut plans = Vec::with_capacity(9);
    plans.push(bundle_plan(
        args,
        &cdn_tree,
        &qwen_manifest,
        "shared-qwen3-vl",
    )?);
    plans.push(bundle_plan(
        args,
        &cdn_tree,
        &q4_qwen_manifest,
        "shared-qwen3-vl-packed-q4s",
    )?);
    plans.push(bundle_plan(
        args,
        &cdn_tree,
        &vae_manifest,
        "shared-flux-vae",
    )?);
    for manifest in &pipeline_manifests {
        plans.push(bundle_plan(args, &cdn_tree, manifest, "boogu-pipeline")?);
    }
    for manifest in &q4_manifests {
        plans.push(bundle_plan(
            args,
            &cdn_tree,
            manifest,
            "boogu-packed-q4s-pipeline",
        )?);
    }

    let component_ids = vec![
        QWEN_COMPONENT_BUNDLE_ID.to_owned(),
        QWEN_Q4S_COMPONENT_BUNDLE_ID.to_owned(),
        FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID.to_owned(),
    ];
    let pipeline_ids = BUNDLES
        .iter()
        .map(|spec| spec.canonical_id.to_owned())
        .chain(Q4_BUNDLES.iter().map(|spec| spec.canonical_id.to_owned()))
        .collect::<Vec<_>>();
    let plan = UploadPlan {
        schema_version: 3,
        release: "boogu-image-0.1-production-with-packed-q4s",
        cdn_root: CDN_ROOT,
        generated_from: path_text(&args.artifact_root),
        cache_control: CACHE_CONTROL,
        dependency_first: true,
        manifest_last: true,
        bundle_count: plans.len(),
        bundles: plans,
        upload_phases: vec![
            UploadPhase {
                sequence: 1,
                name: "dependency-payloads",
                bundles: component_ids.clone(),
                include: "all files except manifest.json",
                cache_control: CACHE_CONTROL,
                prerequisite: None,
            },
            UploadPhase {
                sequence: 2,
                name: "dependency-manifests",
                bundles: component_ids,
                include: "manifest.json only",
                cache_control: "no-cache",
                prerequisite: Some("dependency-payloads"),
            },
            UploadPhase {
                sequence: 3,
                name: "pipeline-payloads",
                bundles: pipeline_ids.clone(),
                include: "all files except manifest.json",
                cache_control: CACHE_CONTROL,
                prerequisite: Some("dependency-manifests"),
            },
            UploadPhase {
                sequence: 4,
                name: "pipeline-manifests",
                bundles: pipeline_ids,
                include: "manifest.json only",
                cache_control: "no-cache",
                prerequisite: Some("pipeline-payloads"),
            },
        ],
    };
    finalize_output(
        args,
        output_stage,
        cleanup,
        plan,
        "verification-reports/modular-equivalence.json",
        duplicate_shared_bytes_removed,
    )
}

fn finalize_output(
    args: &Args,
    output_stage: PathBuf,
    mut cleanup: Cleanup,
    plan: UploadPlan,
    evidence_report: &str,
    duplicate_shared_bytes_removed: u64,
) -> Result<(), Box<dyn Error>> {
    let plan_bytes = json_bytes(&plan)?;
    let plan_sha256 = sha256_bytes(&plan_bytes);
    write_bytes_new(&output_stage.join("upload-plan.json"), &plan_bytes)?;
    write_bytes_new(
        &output_stage.join("upload-plan.json.sha256"),
        format!("{plan_sha256}  upload-plan.json\n").as_bytes(),
    )?;

    fs::rename(&output_stage, &args.output_root)?;
    cleanup.armed = false;
    let absolute_output = fs::canonicalize(&args.output_root)?;
    let logical_declared_payload_bytes = plan.bundles.iter().try_fold(0_u64, |sum, bundle| {
        sum.checked_add(bundle.payload_bytes)
            .ok_or("logical payload byte count overflow")
    })?;
    let physical_transport_parts = plan.bundles.iter().try_fold(0_usize, |sum, bundle| {
        sum.checked_add(bundle.transport_parts)
            .ok_or("transport part count overflow")
    })?;
    let physical_transport_payload_bytes = plan.bundles.iter().try_fold(0_u64, |sum, bundle| {
        sum.checked_add(bundle.transport_payload_bytes)
            .ok_or("transport payload byte count overflow")
    })?;
    let largest_transport_part_bytes = plan
        .bundles
        .iter()
        .map(|bundle| bundle.largest_transport_part_bytes)
        .max()
        .unwrap_or(0);
    let summary = PreparationSummary {
        output_root: path_text(&absolute_output),
        upload_tree: path_text(&absolute_output.join("aberration.technology/model")),
        upload_plan: path_text(&absolute_output.join("upload-plan.json")),
        upload_plan_sha256: plan_sha256,
        equivalence_report: path_text(&absolute_output.join(evidence_report)),
        bundles: plan.bundles.len(),
        logical_declared_payload_bytes,
        physical_transport_parts,
        physical_transport_payload_bytes,
        largest_transport_part_bytes,
        duplicate_shared_bytes_removed,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn prepare_q4_only(
    args: &Args,
    output_stage: PathBuf,
    cdn_tree: PathBuf,
    reports_root: PathBuf,
    cleanup: Cleanup,
    sources: &[SourceBundle],
) -> Result<(), Box<dyn Error>> {
    prove_shared_inputs(sources)?;
    prove_denoisers_disjoint(sources)?;
    let reference = sources.first().ok_or("no Q4 source bundles")?;
    let qwen_manifest = stage_component_bundle(
        &cdn_tree,
        reference,
        ComponentSpec {
            owner: Owner::Qwen,
            bundle: QWEN_Q4S_COMPONENT_BUNDLE_ID,
            profile: QWEN_Q4S_BASE_CONDITIONING_PROFILE,
            model: QWEN_COMPONENT_MODEL_ID,
            model_revision: QWEN_COMPONENT_MODEL_REVISION,
            numeric_format: NumericFormat::Other(QWEN_Q4S_BASE_CONDITIONING_PROFILE.to_owned()),
            bind_source_bundle: true,
        },
        args.copy,
    )?;
    let vae_manifest = stage_component_bundle(
        &cdn_tree,
        reference,
        ComponentSpec {
            owner: Owner::Vae,
            bundle: FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID,
            profile: FLUX_VAE_COMPONENT_PROFILE,
            model: FLUX_VAE_COMPONENT_MODEL_ID,
            model_revision: FLUX_VAE_COMPONENT_MODEL_REVISION,
            numeric_format: NumericFormat::F16,
            bind_source_bundle: false,
        },
        args.copy,
    )?;
    let qwen_dependency = dependency_from_manifest(QWEN_COMPONENT_ROLE, &qwen_manifest)?;
    let vae_dependency = dependency_from_manifest(FLUX_VAE_COMPONENT_ROLE, &vae_manifest)?;
    let manifests = sources
        .iter()
        .map(|source| {
            stage_pipeline_bundle(
                &cdn_tree,
                source,
                &[qwen_dependency.clone(), vae_dependency.clone()],
                args.copy,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let all_manifests = std::iter::once(&qwen_manifest)
        .chain(std::iter::once(&vae_manifest))
        .chain(manifests.iter())
        .collect::<Vec<_>>();
    let digest_map = all_manifests
        .iter()
        .map(|manifest| (manifest.bundle.to_string(), sealed_digest(manifest)))
        .collect::<BTreeMap<_, _>>();
    write_json_new(
        &reports_root.join("candidate-manifest-digests.json"),
        &digest_map,
    )?;
    eprintln!(
        "candidate transport-sealed manifest digests:\n{}",
        serde_json::to_string_pretty(&digest_map)?
    );
    let resolved = all_manifests
        .iter()
        .map(|manifest| (manifest.bundle.clone(), *manifest))
        .collect::<BTreeMap<_, _>>();
    for (source, manifest) in sources.iter().zip(&manifests) {
        manifest.validate_dependency_closure(|bundle| resolved.get(bundle).copied())?;
        prove_reconstructed_closure(source, manifest, &qwen_manifest, &vae_manifest, &cdn_tree)?;
        let verification = verify_modular_release_artifact_directories(
            cdn_tree.join(manifest.bundle.as_str()),
            cdn_tree.join(qwen_manifest.bundle.as_str()),
            cdn_tree.join(vae_manifest.bundle.as_str()),
        )?;
        require_matching_q4_modular_verification(&source.verification, &verification)?;
        write_json_new(
            &reports_root.join(format!("{}-semantic.json", manifest.bundle)),
            &verification,
        )?;
    }
    if fs::read_dir(&cdn_tree)?.count() != Q4_BUNDLES.len() + 2 {
        return Err(
            "Q4-only CDN tree must contain one shared Qwen component, one shared VAE component, and three denoiser parents"
                .into(),
        );
    }
    let component_ids = vec![
        qwen_manifest.bundle.to_string(),
        vae_manifest.bundle.to_string(),
    ];
    let pipeline_ids = manifests
        .iter()
        .map(|manifest| manifest.bundle.to_string())
        .collect::<Vec<_>>();
    let mut bundles = Vec::with_capacity(5);
    bundles.push(bundle_plan(
        args,
        &cdn_tree,
        &qwen_manifest,
        "shared-qwen3-vl-packed-q4s",
    )?);
    bundles.push(bundle_plan(
        args,
        &cdn_tree,
        &vae_manifest,
        "shared-flux-vae",
    )?);
    for manifest in &manifests {
        bundles.push(bundle_plan(
            args,
            &cdn_tree,
            manifest,
            "boogu-packed-q4s-pipeline",
        )?);
    }
    let qwen_bytes = weight_bytes(&qwen_manifest)?;
    let vae_bytes = weight_bytes(&vae_manifest)?;
    let duplicate_shared_bytes_removed = qwen_bytes
        .checked_add(vae_bytes)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or("shared byte count overflow")?;
    let proof = SharedContractProof {
        schema_version: 2,
        source_bundles: sources.iter().map(|source| source.id.clone()).collect(),
        qwen_declarations_identical: true,
        qwen_upstream_sources_identical: true,
        qwen_weight_objects: weight_count(&qwen_manifest),
        qwen_weight_bytes: qwen_bytes,
        vae_declarations_identical: true,
        vae_upstream_source_identical: true,
        vae_config_semantically_identical_after_provenance_normalization: true,
        vae_weight_objects: weight_count(&vae_manifest),
        vae_weight_bytes: vae_bytes,
        denoiser_payloads_pairwise_disjoint: true,
        reconstructed_source_closures_exact: true,
        dependency_closures_verified: true,
        component_contracts_verified: true,
        bounded_burnpacks_verified: true,
        bounded_transport_parts_verified: true,
        transport_part_target_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
        maximum_transport_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        duplicate_shared_bytes_removed,
        component_revision_algorithm: "sha256(compact-json(owner-filtered-source-files-sorted-by-path) + LF)",
        qwen_manifest_digest: sealed_digest(&qwen_manifest),
        q4_qwen_manifest_digest: None,
        vae_manifest_digest: sealed_digest(&vae_manifest),
        pipeline_manifest_digests: manifests
            .iter()
            .map(|manifest| (manifest.bundle.to_string(), sealed_digest(manifest)))
            .collect(),
    };
    write_json_new(&reports_root.join("modular-equivalence.json"), &proof)?;
    let plan = UploadPlan {
        schema_version: 3,
        release: "boogu-image-0.1-packed-q4s",
        cdn_root: CDN_ROOT,
        generated_from: path_text(&args.artifact_root),
        cache_control: CACHE_CONTROL,
        dependency_first: true,
        manifest_last: true,
        bundle_count: bundles.len(),
        bundles,
        upload_phases: vec![
            UploadPhase {
                sequence: 1,
                name: "dependency-payloads",
                bundles: component_ids.clone(),
                include: "all files except manifest.json",
                cache_control: CACHE_CONTROL,
                prerequisite: None,
            },
            UploadPhase {
                sequence: 2,
                name: "dependency-manifests",
                bundles: component_ids,
                include: "manifest.json only",
                cache_control: "no-cache",
                prerequisite: Some("dependency-payloads"),
            },
            UploadPhase {
                sequence: 3,
                name: "pipeline-payloads",
                bundles: pipeline_ids.clone(),
                include: "all files except manifest.json",
                cache_control: CACHE_CONTROL,
                prerequisite: Some("dependency-manifests"),
            },
            UploadPhase {
                sequence: 4,
                name: "pipeline-manifests",
                bundles: pipeline_ids,
                include: "manifest.json only",
                cache_control: "no-cache",
                prerequisite: Some("pipeline-payloads"),
            },
        ],
    };
    finalize_output(
        args,
        output_stage,
        cleanup,
        plan,
        "verification-reports/modular-equivalence.json",
        duplicate_shared_bytes_removed,
    )
}

fn require_matching_q4_modular_verification(
    source: &BooguReleaseVerification,
    candidate: &burn_boogu::artifacts::BooguModularReleaseVerification,
) -> Result<(), Box<dyn Error>> {
    if candidate.variant != source.variant
        || candidate.profile != source.profile
        || candidate.verified_weight_objects != source.verified_weight_objects
        || candidate.verified_tensors != source.verified_tensors
        || candidate.largest_object_bytes != source.largest_object_bytes
        || candidate.parent.max_shard_bytes != source.max_shard_bytes
    {
        return Err(
            "Q4 transport candidate differs from its fully verified semantic source".into(),
        );
    }
    Ok(())
}

fn read_and_verify_source_bundles(args: &Args) -> Result<Vec<SourceBundle>, Box<dyn Error>> {
    BUNDLES
        .iter()
        .copied()
        .map(|spec| {
            let profile = BooguStorageProfile::F16QwenVisionF32;
            let id = source_artifact_bundle_id(spec.variant, profile);
            let directory = args.artifact_root.join(&id);
            let manifest = read_exact_source_manifest(&directory, spec, profile, &id)?;
            // This is the expensive gate: parse configs/inventories and authenticate every bounded
            // Burnpack from the exact conversion source before any file is promoted.
            let verification = verify_release_artifact_directory(&directory)?;
            let inventory = read_json_array(&directory.join(INVENTORY_PATH))?;
            let source_files = read_json_array(&directory.join(SOURCE_FILES_PATH))?;
            let normalized_vae_config = normalized_vae_config(&directory.join(VAE_CONFIG_PATH))?;
            Ok(SourceBundle {
                spec,
                id,
                directory,
                manifest,
                verification,
                inventory,
                source_files,
                normalized_vae_config,
            })
        })
        .collect()
}

fn read_and_verify_q4_bundles(args: &Args) -> Result<Vec<SourceBundle>, Box<dyn Error>> {
    Q4_BUNDLES
        .iter()
        .copied()
        .map(|spec| {
            let profile = BooguStorageProfile::Q4sBlockUpTo128F32;
            let id = source_artifact_bundle_id(spec.variant, profile);
            let directory = args.artifact_root.join(&id);
            let manifest = read_exact_source_manifest(&directory, spec, profile, &id)?;
            let verification = verify_release_artifact_directory(&directory)?;
            let inventory = read_json_array(&directory.join(INVENTORY_PATH))?;
            let source_files = read_json_array(&directory.join(SOURCE_FILES_PATH))?;
            let normalized_vae_config = normalized_vae_config(&directory.join(VAE_CONFIG_PATH))?;
            Ok(SourceBundle {
                spec,
                id,
                directory,
                manifest,
                verification,
                inventory,
                source_files,
                normalized_vae_config,
            })
        })
        .collect()
}

fn read_exact_source_manifest(
    source_dir: &Path,
    spec: BundleSpec,
    profile: BooguStorageProfile,
    expected_bundle: &str,
) -> Result<ArtifactManifest, Box<dyn Error>> {
    require_real_directory(source_dir)?;
    require_regular_path(source_dir, Path::new("manifest.json"))?;
    let manifest: ArtifactManifest =
        serde_json::from_slice(&fs::read(source_dir.join("manifest.json"))?)?;
    manifest.validate_sealed()?;
    if manifest.schema_version != 1 || !manifest.dependencies.is_empty() {
        return Err(format!("source {expected_bundle} is not dependency-free schema v1").into());
    }
    if manifest.bundle.as_str() != expected_bundle {
        return Err(format!(
            "source bundle {} differs from expected {expected_bundle}",
            manifest.bundle
        )
        .into());
    }
    let actual = sealed_digest(&manifest);
    let expected = release_source_artifact_digest(spec.variant, profile)
        .ok_or("release tuple has no pinned source digest")?;
    if actual != expected {
        return Err(format!(
            "source {expected_bundle} digest {actual} differs from pinned {expected}"
        )
        .into());
    }
    let (expected_converter, expected_profile) = match profile {
        BooguStorageProfile::F16QwenVisionF32 => ("0.1.0", "f16-qwen-vision-f32"),
        BooguStorageProfile::Q4sBlockUpTo128F32 => {
            (env!("CARGO_PKG_VERSION"), "q4s-block-up-to128-f32")
        }
        _ => return Err("CDN preparation received an unsupported source profile".into()),
    };
    if manifest
        .metadata
        .get("conversion_crate")
        .map(String::as_str)
        != Some(expected_converter)
        || manifest.profile.as_str() != expected_profile
    {
        return Err(format!(
            "source {expected_bundle} has the wrong converter/profile; expected {expected_converter}/{expected_profile}"
        )
        .into());
    }
    let descriptor = boogu_model_descriptor(spec.variant);
    if manifest.model != descriptor.id || manifest.model_revision != descriptor.revision {
        return Err(format!("source {expected_bundle} has the wrong model identity").into());
    }
    Ok(manifest)
}

fn prove_shared_inputs(sources: &[SourceBundle]) -> Result<(), Box<dyn Error>> {
    let reference = sources.first().ok_or("no source bundles")?;
    let qwen_files = owner_weight_files(&reference.manifest, Owner::Qwen);
    let vae_files = owner_weight_files(&reference.manifest, Owner::Vae);
    let qwen_inventory = inventory_for(&reference.inventory, Owner::Qwen)?;
    let vae_inventory = inventory_for(&reference.inventory, Owner::Vae)?;
    let qwen_sources = source_files_for(&reference.source_files, "mllm/")?;
    let vae_sources = source_files_for(&reference.source_files, "vae/")?;
    if source_contract_revision(&qwen_sources)? != QWEN_COMPONENT_MODEL_REVISION
        || source_contract_revision(&vae_sources)? != FLUX_VAE_COMPONENT_MODEL_REVISION
    {
        return Err(
            "component model revision does not match its canonical source declarations".into(),
        );
    }
    let qwen_metadata = compact_files_for(&reference.manifest, Owner::Qwen)?;

    for candidate in &sources[1..] {
        if owner_weight_files(&candidate.manifest, Owner::Qwen) != qwen_files
            || inventory_for(&candidate.inventory, Owner::Qwen)? != qwen_inventory
            || source_files_for(&candidate.source_files, "mllm/")? != qwen_sources
            || compact_files_for(&candidate.manifest, Owner::Qwen)? != qwen_metadata
        {
            return Err(format!("{} does not share the exact Qwen contract", candidate.id).into());
        }
        if owner_weight_files(&candidate.manifest, Owner::Vae) != vae_files
            || inventory_for(&candidate.inventory, Owner::Vae)? != vae_inventory
            || source_files_for(&candidate.source_files, "vae/")? != vae_sources
            || candidate.normalized_vae_config != reference.normalized_vae_config
        {
            return Err(format!(
                "{} does not share the exact FLUX VAE contract",
                candidate.id
            )
            .into());
        }
    }
    Ok(())
}

fn prove_cross_profile_vae(
    mixed_f16: &SourceBundle,
    packed_q4s: &SourceBundle,
) -> Result<(), Box<dyn Error>> {
    if owner_weight_files(&mixed_f16.manifest, Owner::Vae)
        != owner_weight_files(&packed_q4s.manifest, Owner::Vae)
        || inventory_for(&mixed_f16.inventory, Owner::Vae)?
            != inventory_for(&packed_q4s.inventory, Owner::Vae)?
        || source_files_for(&mixed_f16.source_files, "vae/")?
            != source_files_for(&packed_q4s.source_files, "vae/")?
        || mixed_f16.normalized_vae_config != packed_q4s.normalized_vae_config
    {
        return Err(
            "mixed-F16 and packed-Q4S releases do not share the exact FLUX VAE contract".into(),
        );
    }
    Ok(())
}

fn prove_denoisers_disjoint(sources: &[SourceBundle]) -> Result<(), Box<dyn Error>> {
    let sets = sources
        .iter()
        .map(|bundle| {
            owner_weight_files(&bundle.manifest, Owner::Denoiser)
                .into_iter()
                .map(|file| file.sha256)
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    for left in 0..sets.len() {
        for right in (left + 1)..sets.len() {
            if let Some(shared) = sets[left].intersection(&sets[right]).next() {
                return Err(format!(
                    "denoiser payloads for {} and {} overlap at {shared}",
                    sources[left].id, sources[right].id
                )
                .into());
            }
        }
    }
    Ok(())
}

struct ComponentSpec {
    owner: Owner,
    bundle: &'static str,
    profile: &'static str,
    model: &'static str,
    model_revision: &'static str,
    numeric_format: NumericFormat,
    bind_source_bundle: bool,
}

fn stage_component_bundle(
    cdn_tree: &Path,
    source: &SourceBundle,
    spec: ComponentSpec,
    copy: bool,
) -> Result<ArtifactManifest, Box<dyn Error>> {
    let destination = cdn_tree.join(spec.bundle);
    fs::create_dir(&destination)?;
    let mut files = owner_weight_files(&source.manifest, spec.owner);
    let mut compact = compact_files_for(&source.manifest, spec.owner)?;
    for file in files.iter().chain(compact.iter()) {
        materialize_file(&source.directory, &destination, file, copy)?;
    }
    files.append(&mut compact);

    let inventory = inventory_for(&source.inventory, spec.owner)?;
    files.push(write_json_artifact(
        &destination,
        INVENTORY_PATH,
        &Value::Array(inventory.clone()),
        ArtifactFileRole::Metadata,
    )?);
    let source_prefix = match spec.owner {
        Owner::Qwen => "mllm/",
        Owner::Vae => "vae/",
        Owner::Denoiser => unreachable!("denoiser is staged as a pipeline"),
    };
    files.push(write_json_artifact(
        &destination,
        SOURCE_FILES_PATH,
        &Value::Array(source_files_for(&source.source_files, source_prefix)?),
        ArtifactFileRole::Metadata,
    )?);
    if spec.owner == Owner::Vae {
        // `_name_or_path` is workstation provenance, not model semantics.  The normalized config
        // is a new file; never truncate the source through a hardlink.
        files.push(write_json_artifact(
            &destination,
            VAE_CONFIG_PATH,
            &source.normalized_vae_config,
            ArtifactFileRole::Config,
        )?);
    }

    let components = components_for(&source.manifest, spec.owner);
    let mut metadata = component_metadata(source, spec.owner, &inventory, spec.bind_source_bundle);
    metadata.insert("component_bundle".into(), "true".into());
    metadata.insert(
        "component_kind".into(),
        match spec.owner {
            Owner::Qwen => "qwen3-vl-base-conditioning",
            Owner::Vae => "flux1-vae",
            Owner::Denoiser => unreachable!("denoiser is a composed parent"),
        }
        .into(),
    );
    let mut manifest = ArtifactManifest {
        schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
        bundle: ArtifactBundleId::new(spec.bundle)?,
        profile: ArtifactProfileId::new(spec.profile)?,
        model: ModelId::new(spec.model)?,
        model_revision: spec.model_revision.to_owned(),
        numeric_format: spec.numeric_format,
        components,
        files,
        dependencies: Vec::new(),
        metadata,
        content_digest: None,
    };
    manifest.seal()?;
    verify_materialized_manifest(&destination, &manifest, copy)?;
    install_transport_layout(&destination, &mut manifest)?;
    write_manifest(&destination, &manifest)?;
    Ok(manifest)
}

fn stage_pipeline_bundle(
    cdn_tree: &Path,
    source: &SourceBundle,
    dependencies: &[ArtifactDependency],
    copy: bool,
) -> Result<ArtifactManifest, Box<dyn Error>> {
    let destination = cdn_tree.join(source.spec.canonical_id);
    fs::create_dir(&destination)?;
    let mut files = owner_weight_files(&source.manifest, Owner::Denoiser);
    let mut compact = compact_files_for(&source.manifest, Owner::Denoiser)?;
    for file in files.iter().chain(compact.iter()) {
        materialize_file(&source.directory, &destination, file, copy)?;
    }
    files.append(&mut compact);
    let inventory = inventory_for(&source.inventory, Owner::Denoiser)?;
    files.push(write_json_artifact(
        &destination,
        INVENTORY_PATH,
        &Value::Array(inventory.clone()),
        ArtifactFileRole::Metadata,
    )?);
    files.push(write_json_artifact(
        &destination,
        SOURCE_FILES_PATH,
        &Value::Array(source_files_for(&source.source_files, "transformer/")?),
        ArtifactFileRole::Metadata,
    )?);

    let mut metadata = component_metadata(source, Owner::Denoiser, &inventory, true);
    metadata.insert("composition_manifest".into(), "true".into());
    metadata.insert(
        "artifact_layout".into(),
        "semantic-burnpack-composition-v2".into(),
    );
    metadata.insert(
        "component_dependency_count".into(),
        dependencies.len().to_string(),
    );
    metadata.insert("algorithm".into(), "dmd-turbo".into());
    metadata.insert("profile".into(), source.manifest.profile.to_string());
    let mut manifest = ArtifactManifest {
        schema_version: ARTIFACT_MANIFEST_SCHEMA_V2,
        bundle: ArtifactBundleId::new(source.spec.canonical_id)?,
        profile: source.manifest.profile.clone(),
        model: source.manifest.model.clone(),
        model_revision: source.manifest.model_revision.clone(),
        numeric_format: source.manifest.numeric_format.clone(),
        components: components_for(&source.manifest, Owner::Denoiser),
        files,
        dependencies: dependencies.to_vec(),
        metadata,
        content_digest: None,
    };
    manifest.seal()?;
    verify_materialized_manifest(&destination, &manifest, copy)?;
    install_transport_layout(&destination, &mut manifest)?;
    write_manifest(&destination, &manifest)?;
    Ok(manifest)
}

fn component_metadata(
    source: &SourceBundle,
    owner: Owner,
    inventory: &[Value],
    bind_source_bundle: bool,
) -> BTreeMap<String, String> {
    let included = inventory
        .iter()
        .filter(|entry| {
            entry
                .get("included")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        })
        .count();
    let mut metadata = BTreeMap::new();
    metadata.insert("artifact_layout".into(), "semantic-burnpack-v1".into());
    metadata.insert("conversion_crate".into(), "0.1.0".into());
    metadata.insert("layout_contract".into(), INVENTORY_PATH.into());
    metadata.insert("owner".into(), owner.inventory_name().into());
    metadata.insert("physical_shards_bounded".into(), "true".into());
    metadata.insert("oversized_tensor_shards".into(), "0".into());
    metadata.insert(
        "source_revision".into(),
        "25f8f888298224a94e5ec2abafb98abea9031a0d".into(),
    );
    metadata.insert("stored_tensor_count".into(), included.to_string());
    metadata.insert(
        "tensor_inventory_entries".into(),
        inventory.len().to_string(),
    );
    metadata.insert("tensor_inventory_schema".into(), "2".into());
    metadata.insert(
        ARTIFACT_TARGET_MAX_SEMANTIC_SHARD_BYTES_KEY.into(),
        ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
    );
    if bind_source_bundle {
        metadata.insert(
            "verified_source_bundle".into(),
            source.manifest.bundle.to_string(),
        );
        metadata.insert(
            "verified_source_digest".into(),
            sealed_digest(&source.manifest),
        );
    }
    if owner == Owner::Qwen {
        metadata.insert("omitted_tensor_count".into(), "1".into());
        metadata.insert("qwen_embedding_row_chunks".into(), "6".into());
        metadata.insert("qwen_lm_head".into(), "omitted-base-model".into());
        metadata.insert("tensor_count".into(), "749".into());
    } else {
        metadata.insert("omitted_tensor_count".into(), "0".into());
        metadata.insert("tensor_count".into(), inventory.len().to_string());
    }
    metadata
}

fn prove_reconstructed_closure(
    source: &SourceBundle,
    pipeline: &ArtifactManifest,
    qwen: &ArtifactManifest,
    vae: &ArtifactManifest,
    cdn_tree: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut reconstructed_weights = qwen
        .files
        .iter()
        .chain(vae.files.iter())
        .chain(pipeline.files.iter())
        .filter(|file| file.role == ArtifactFileRole::Weights)
        .cloned()
        .collect::<Vec<_>>();
    let mut original_weights = source
        .manifest
        .files
        .iter()
        .filter(|file| file.role == ArtifactFileRole::Weights)
        .cloned()
        .collect::<Vec<_>>();
    sort_files(&mut reconstructed_weights);
    sort_files(&mut original_weights);
    if reconstructed_weights != original_weights {
        return Err(format!(
            "{} modular weights do not reconstruct the flat contract",
            source.id
        )
        .into());
    }

    let original_compact = source
        .manifest
        .files
        .iter()
        .filter(|file| {
            file.role != ArtifactFileRole::Weights
                && !matches!(
                    file.path.as_str(),
                    INVENTORY_PATH
                        | SOURCE_FILES_PATH
                        | VAE_CONFIG_PATH
                        | ARTIFACT_TRANSPORT_LAYOUT_PATH
                )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut reconstructed_compact = qwen
        .files
        .iter()
        .chain(vae.files.iter())
        .chain(pipeline.files.iter())
        .filter(|file| {
            file.role != ArtifactFileRole::Weights
                && !matches!(
                    file.path.as_str(),
                    INVENTORY_PATH
                        | SOURCE_FILES_PATH
                        | VAE_CONFIG_PATH
                        | ARTIFACT_TRANSPORT_LAYOUT_PATH
                )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut original_compact = original_compact;
    sort_files(&mut original_compact);
    sort_files(&mut reconstructed_compact);
    if reconstructed_compact != original_compact {
        return Err(format!(
            "{} compact files are not classified exactly once",
            source.id
        )
        .into());
    }

    let mut reconstructed_inventory = Vec::new();
    reconstructed_inventory.extend(read_json_array(
        &cdn_tree.join(qwen.bundle.as_str()).join(INVENTORY_PATH),
    )?);
    reconstructed_inventory.extend(read_json_array(
        &cdn_tree.join(vae.bundle.as_str()).join(INVENTORY_PATH),
    )?);
    reconstructed_inventory.extend(read_json_array(
        &cdn_tree.join(pipeline.bundle.as_str()).join(INVENTORY_PATH),
    )?);
    require_json_multiset_equal(
        &source.inventory,
        &reconstructed_inventory,
        "tensor inventory",
    )?;

    let mut reconstructed_sources = Vec::new();
    reconstructed_sources.extend(read_json_array(
        &cdn_tree.join(qwen.bundle.as_str()).join(SOURCE_FILES_PATH),
    )?);
    reconstructed_sources.extend(read_json_array(
        &cdn_tree.join(vae.bundle.as_str()).join(SOURCE_FILES_PATH),
    )?);
    reconstructed_sources.extend(read_json_array(
        &cdn_tree
            .join(pipeline.bundle.as_str())
            .join(SOURCE_FILES_PATH),
    )?);
    require_json_multiset_equal(&source.source_files, &reconstructed_sources, "source files")?;

    let staged_vae =
        normalized_vae_config(&cdn_tree.join(vae.bundle.as_str()).join(VAE_CONFIG_PATH))?;
    if staged_vae != source.normalized_vae_config {
        return Err("normalized VAE config differs from the source semantic config".into());
    }
    let mut original_components = source.manifest.components.clone();
    let mut reconstructed_components = qwen
        .components
        .iter()
        .chain(vae.components.iter())
        .chain(pipeline.components.iter())
        .cloned()
        .collect::<Vec<_>>();
    original_components.sort_by(|left, right| left.id.cmp(&right.id));
    reconstructed_components.sort_by(|left, right| left.id.cmp(&right.id));
    if original_components != reconstructed_components {
        return Err("component closure does not reconstruct the source component set".into());
    }
    Ok(())
}

fn owner_weight_files(manifest: &ArtifactManifest, owner: Owner) -> Vec<ArtifactFile> {
    manifest
        .files
        .iter()
        .filter(|file| {
            file.role == ArtifactFileRole::Weights
                && file
                    .component
                    .as_ref()
                    .is_some_and(|component| owner.owns_component(component.as_str()))
        })
        .cloned()
        .collect()
}

fn compact_files_for(
    manifest: &ArtifactManifest,
    owner: Owner,
) -> Result<Vec<ArtifactFile>, Box<dyn Error>> {
    let mut files = Vec::new();
    for file in manifest
        .files
        .iter()
        .filter(|file| file.role != ArtifactFileRole::Weights)
    {
        let path = file.path.as_str();
        let selected = match owner {
            Owner::Qwen => {
                path.starts_with("metadata/source/mllm/")
                    || path.starts_with("metadata/source/processor/")
            }
            Owner::Vae => false, // the VAE config is normalized into a new, non-hardlinked file
            Owner::Denoiser => {
                path == "metadata/source/model_index.json"
                    || path.starts_with("metadata/source/scheduler/")
                    || path.starts_with("metadata/source/transformer/")
            }
        };
        if selected {
            files.push(file.clone());
        } else if !matches!(path, INVENTORY_PATH | SOURCE_FILES_PATH | VAE_CONFIG_PATH)
            && !path.starts_with("metadata/source/mllm/")
            && !path.starts_with("metadata/source/processor/")
            && path != "metadata/source/model_index.json"
            && !path.starts_with("metadata/source/scheduler/")
            && !path.starts_with("metadata/source/transformer/")
        {
            return Err(format!("unclassified compact artifact path {path}").into());
        }
    }
    Ok(files)
}

fn components_for(manifest: &ArtifactManifest, owner: Owner) -> Vec<ArtifactComponent> {
    manifest
        .components
        .iter()
        .filter(|component| owner.owns_component(component.id.as_str()))
        .cloned()
        .collect()
}

fn inventory_for(inventory: &[Value], owner: Owner) -> Result<Vec<Value>, Box<dyn Error>> {
    inventory
        .iter()
        .filter_map(|entry| {
            let actual = entry.get("owner").and_then(Value::as_str);
            match actual {
                Some(actual) if actual == owner.inventory_name() => Some(Ok(entry.clone())),
                Some("qwen3-vl" | "boogu-denoiser" | "flux-vae") => None,
                Some(actual) => Some(Err(format!("unknown tensor owner {actual}").into())),
                None => Some(Err("tensor inventory entry has no owner".into())),
            }
        })
        .collect()
}

fn source_files_for(source_files: &[Value], prefix: &str) -> Result<Vec<Value>, Box<dyn Error>> {
    source_files
        .iter()
        .filter_map(|entry| match entry.get("path").and_then(Value::as_str) {
            Some(path) if path.starts_with(prefix) => Some(Ok(entry.clone())),
            Some(_) => None,
            None => Some(Err("source-files entry has no path".into())),
        })
        .collect()
}

fn source_contract_revision(records: &[Value]) -> Result<String, Box<dyn Error>> {
    #[derive(Serialize)]
    struct CanonicalSourceFile<'a> {
        path: &'a str,
        size: u64,
        sha256: &'a str,
    }

    let mut records = records
        .iter()
        .map(|record| {
            Ok(CanonicalSourceFile {
                path: record
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("source declaration omits path")?,
                size: record
                    .get("size")
                    .and_then(Value::as_u64)
                    .ok_or("source declaration omits size")?,
                sha256: record
                    .get("sha256")
                    .and_then(Value::as_str)
                    .ok_or("source declaration omits sha256")?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    records.sort_by_key(|record| record.path);
    let mut bytes = serde_json::to_vec(&records)?;
    bytes.push(b'\n');
    Ok(sha256_bytes(&bytes))
}

fn normalized_vae_config(path: &Path) -> Result<Value, Box<dyn Error>> {
    let mut value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let object = value
        .as_object_mut()
        .ok_or("VAE config must be a JSON object")?;
    object.remove("_name_or_path");
    Ok(value)
}

fn read_json_array(path: &Path) -> Result<Vec<Value>, Box<dyn Error>> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    value
        .as_array()
        .cloned()
        .ok_or_else(|| format!("{} must be a JSON array", path.display()).into())
}

fn require_json_multiset_equal(
    expected: &[Value],
    actual: &[Value],
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let mut expected = expected
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
    let mut actual = actual
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
    expected.sort();
    actual.sort();
    if expected != actual {
        return Err(format!("reconstructed {label} differs from the source contract").into());
    }
    Ok(())
}

fn write_json_artifact(
    root: &Path,
    relative: &str,
    value: &Value,
    role: ArtifactFileRole,
) -> Result<ArtifactFile, Box<dyn Error>> {
    let path = ArtifactPath::new(relative)?;
    let bytes = json_bytes(value)?;
    let destination = root.join(relative);
    fs::create_dir_all(destination.parent().ok_or("metadata path has no parent")?)?;
    write_bytes_new(&destination, &bytes)?;
    Ok(ArtifactFile {
        path,
        size: u64::try_from(bytes.len())?,
        sha256: Sha256Digest::calculate(&bytes),
        role,
        component: None,
        shard: None,
    })
}

fn install_transport_layout(
    root: &Path,
    manifest: &mut ArtifactManifest,
) -> Result<TransportStats, Box<dyn Error>> {
    if ARTIFACT_TRANSPORT_TARGET_PART_BYTES > ARTIFACT_TRANSPORT_MAX_PART_BYTES {
        return Err("transport part target exceeds the hard physical CDN-object ceiling".into());
    }
    let mut weights = manifest
        .files
        .iter()
        .filter(|file| file.role == ArtifactFileRole::Weights)
        .cloned()
        .collect::<Vec<_>>();
    weights.sort_by(|left, right| left.path.cmp(&right.path));
    if weights.is_empty() {
        return Err("transport layout requires at least one semantic weight object".into());
    }

    let transport_root = root.join("transport");
    fs::create_dir(&transport_root)?;
    let mut objects = Vec::with_capacity(weights.len());
    let mut unique_parts = BTreeMap::<ArtifactPath, (u64, Sha256Digest)>::new();
    for file in &weights {
        require_regular_path(root, Path::new(file.path.as_str()))?;
        let source_path = root.join(file.path.as_str());
        let source_size = fs::metadata(&source_path)?.len();
        if source_size != file.size {
            return Err(format!(
                "semantic object {} is {source_size} bytes, expected {}",
                file.path, file.size
            )
            .into());
        }
        if file.size > ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES {
            return Err(format!(
                "semantic object {} is {} bytes, exceeding {ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES}",
                file.path, file.size
            )
            .into());
        }

        let mut input = fs::File::open(&source_path)?;
        let mut offset = 0_u64;
        let mut parts = Vec::new();
        while offset < file.size {
            let expected = ARTIFACT_TRANSPORT_TARGET_PART_BYTES.min(file.size - offset);
            let capacity = usize::try_from(expected)?;
            let mut bytes = vec![0_u8; capacity];
            input.read_exact(&mut bytes)?;
            let digest = Sha256Digest::calculate(&bytes);
            let relative = ArtifactPath::new(format!("transport/{digest}.part"))?;
            let destination = root.join(relative.as_str());
            match fs::symlink_metadata(&destination) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink()
                        || !metadata.is_file()
                        || metadata.len() != expected
                        || sha256_file(&destination)? != digest
                    {
                        return Err(format!(
                            "content-addressed transport part collision at {}",
                            destination.display()
                        )
                        .into());
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    write_bytes_new(&destination, &bytes)?;
                }
                Err(error) => return Err(error.into()),
            }
            if let Some((declared_size, declared_digest)) =
                unique_parts.insert(relative.clone(), (expected, digest))
                && (declared_size != expected || declared_digest != digest)
            {
                return Err(format!("inconsistent transport part declaration {relative}").into());
            }
            parts.push(ArtifactTransportPart {
                path: relative,
                offset,
                size: expected,
                sha256: digest,
            });
            offset = offset
                .checked_add(expected)
                .ok_or("transport part offset overflow")?;
        }
        let mut trailing = [0_u8; 1];
        if input.read(&mut trailing)? != 0 {
            return Err(format!("semantic object {} exceeds its sealed size", file.path).into());
        }
        objects.push(ArtifactTransportObject {
            path: file.path.clone(),
            size: file.size,
            sha256: file.sha256,
            parts,
        });
        fs::remove_file(&source_path)?;
    }

    let layout = ArtifactTransportLayout {
        schema_version: ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION,
        bundle: manifest.bundle.clone(),
        profile: manifest.profile.clone(),
        model: manifest.model.clone(),
        model_revision: manifest.model_revision.clone(),
        target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
        hard_max_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        objects,
    };
    manifest.metadata.insert(
        ARTIFACT_TRANSPORT_LAYOUT_PATH_KEY.into(),
        ARTIFACT_TRANSPORT_LAYOUT_PATH.into(),
    );
    manifest.metadata.insert(
        ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_KEY.into(),
        ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION.to_string(),
    );
    manifest
        .metadata
        .insert(ARTIFACT_TRANSPORT_PARTS_REQUIRED_KEY.into(), "true".into());
    manifest.metadata.insert(
        ARTIFACT_TRANSPORT_PART_TARGET_BYTES_KEY.into(),
        ARTIFACT_TRANSPORT_TARGET_PART_BYTES.to_string(),
    );
    manifest.metadata.insert(
        ARTIFACT_TARGET_MAX_TRANSPORT_SHARD_BYTES_KEY.into(),
        ARTIFACT_TRANSPORT_MAX_PART_BYTES.to_string(),
    );
    manifest.metadata.insert(
        ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES_KEY.into(),
        ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES.to_string(),
    );
    let layout_bytes = json_bytes(&layout)?;
    let layout_path = ArtifactPath::new(ARTIFACT_TRANSPORT_LAYOUT_PATH)?;
    write_bytes_new(&root.join(layout_path.as_str()), &layout_bytes)?;
    manifest.files.push(ArtifactFile {
        path: layout_path,
        size: u64::try_from(layout_bytes.len())?,
        sha256: Sha256Digest::calculate(&layout_bytes),
        role: ArtifactFileRole::Metadata,
        component: None,
        shard: None,
    });
    manifest.seal()?;
    ArtifactTransportLayout::parse_and_validate(manifest, &layout_bytes)?;

    let (bytes, largest_part_bytes) =
        unique_parts
            .values()
            .try_fold((0_u64, 0_u64), |(total, largest), (size, _)| {
                Ok::<_, Box<dyn Error>>((
                    total
                        .checked_add(*size)
                        .ok_or("transport payload byte count overflow")?,
                    largest.max(*size),
                ))
            })?;
    Ok(TransportStats {
        objects: weights.len(),
        parts: unique_parts.len(),
        bytes,
        largest_part_bytes,
        layout_bytes: u64::try_from(layout_bytes.len())?,
    })
}

fn write_manifest(root: &Path, manifest: &ArtifactManifest) -> Result<u64, Box<dyn Error>> {
    let bytes = write_json_new(&root.join("manifest.json"), manifest)?;
    if bytes > MAX_MANIFEST_BYTES {
        return Err(format!(
            "{} manifest exceeds {MAX_MANIFEST_BYTES} bytes",
            manifest.bundle
        )
        .into());
    }
    Ok(bytes)
}

fn verify_materialized_manifest(
    root: &Path,
    manifest: &ArtifactManifest,
    copy: bool,
) -> Result<(), Box<dyn Error>> {
    manifest.validate_sealed()?;
    for file in &manifest.files {
        let relative = Path::new(file.path.as_str());
        require_regular_path(root, relative)?;
        let path = root.join(relative);
        let metadata = fs::metadata(&path)?;
        if metadata.len() != file.size {
            return Err(format!("{} has the wrong size", path.display()).into());
        }
        if copy || !file.path.as_str().starts_with("objects/") {
            let digest = sha256_file(&path)?;
            if digest != file.sha256 {
                return Err(format!("{} has the wrong SHA-256", path.display()).into());
            }
        }
    }
    Ok(())
}

fn bundle_plan(
    args: &Args,
    cdn_tree: &Path,
    manifest: &ArtifactManifest,
    kind: &'static str,
) -> Result<BundlePlan, Box<dyn Error>> {
    let directory = cdn_tree.join(manifest.bundle.as_str());
    let transport = transport_stats(&directory, manifest)?;
    let manifest_bytes = fs::metadata(directory.join("manifest.json"))?.len();
    let payload_bytes = manifest.files.iter().try_fold(0_u64, |sum, file| {
        sum.checked_add(file.size)
            .ok_or("payload byte count overflow")
    })?;
    let largest_payload_bytes = manifest
        .files
        .iter()
        .map(|file| file.size)
        .max()
        .ok_or("manifest has no files")?;
    // `transport_stats` authenticates the sidecar through the shared transport validator, which
    // fail-closes every directly published non-weight file at the physical CDN-object ceiling.
    let browser_fit = transport.largest_part_bytes <= ARTIFACT_TRANSPORT_MAX_PART_BYTES;
    if !browser_fit {
        return Err(format!("{} is not browser transport-fit", manifest.bundle).into());
    }
    Ok(BundlePlan {
        bundle_id: manifest.bundle.to_string(),
        kind,
        profile: manifest.profile.to_string(),
        model: manifest.model.to_string(),
        model_revision: manifest.model_revision.clone(),
        content_digest: sealed_digest(manifest),
        cdn_base_url: format!("{CDN_ROOT}/{}", manifest.bundle),
        manifest_url: format!("{CDN_ROOT}/{}/manifest.json", manifest.bundle),
        local_directory: path_text(
            &args
                .output_root
                .join("aberration.technology/model")
                .join(manifest.bundle.as_str()),
        ),
        files: manifest.files.len(),
        weight_objects: weight_count(manifest),
        payload_bytes,
        largest_payload_bytes,
        transport_objects: transport.objects,
        transport_parts: transport.parts,
        transport_payload_bytes: transport.bytes,
        largest_transport_part_bytes: transport.largest_part_bytes,
        transport_layout_bytes: transport.layout_bytes,
        manifest_bytes,
        dependencies: manifest
            .dependencies
            .iter()
            .map(|dependency| dependency.bundle.to_string())
            .collect(),
        browser_transport_fit: true,
        browser_bounds: BrowserBounds {
            range_chunk_bytes: 4 * 1024 * 1024,
            maximum_response_bytes: 16 * 1024 * 1024,
            maximum_manifest_bytes: MAX_MANIFEST_BYTES,
            maximum_metadata_bytes: MAX_METADATA_BYTES,
            maximum_semantic_object_bytes: ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES,
            transport_part_target_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
            maximum_transport_part_bytes: ARTIFACT_TRANSPORT_MAX_PART_BYTES,
        },
    })
}

fn transport_stats(
    root: &Path,
    manifest: &ArtifactManifest,
) -> Result<TransportStats, Box<dyn Error>> {
    let declaration = ArtifactTransportLayout::declared_file(manifest)?
        .ok_or("manifest does not seal its transport layout")?;
    let relative = declaration.path.as_str();
    require_regular_path(root, Path::new(relative))?;
    let path = root.join(relative);
    if fs::metadata(&path)?.len() != declaration.size {
        return Err("transport layout differs from its sealed declaration".into());
    }
    let bytes = fs::read(path)?;
    let verified = ArtifactTransportLayout::parse_and_validate(manifest, &bytes)?;
    let layout = verified.layout();
    let mut unique_parts = BTreeMap::<ArtifactPath, (u64, Sha256Digest)>::new();
    for object in &layout.objects {
        for part in &object.parts {
            if let Some((size, digest)) =
                unique_parts.insert(part.path.clone(), (part.size, part.sha256))
                && (size != part.size || digest != part.sha256)
            {
                return Err(
                    format!("inconsistent transport part declaration {}", part.path).into(),
                );
            }
            require_regular_path(root, Path::new(part.path.as_str()))?;
            let actual = fs::metadata(root.join(part.path.as_str()))?.len();
            if actual != part.size {
                return Err(format!(
                    "transport part {} is {actual} bytes, expected {}",
                    part.path, part.size
                )
                .into());
            }
        }
    }
    let (payload_bytes, largest_part_bytes) =
        unique_parts
            .values()
            .try_fold((0_u64, 0_u64), |(total, largest), (size, _)| {
                Ok::<_, Box<dyn Error>>((
                    total
                        .checked_add(*size)
                        .ok_or("transport payload byte count overflow")?,
                    largest.max(*size),
                ))
            })?;
    Ok(TransportStats {
        objects: layout.objects.len(),
        parts: unique_parts.len(),
        bytes: payload_bytes,
        largest_part_bytes,
        layout_bytes: declaration.size,
    })
}

fn materialize_file(
    source_root: &Path,
    destination_root: &Path,
    file: &ArtifactFile,
    copy: bool,
) -> Result<(), Box<dyn Error>> {
    let relative = Path::new(file.path.as_str());
    require_regular_path(source_root, relative)?;
    let source = source_root.join(relative);
    let destination = destination_root.join(relative);
    fs::create_dir_all(
        destination
            .parent()
            .ok_or("payload destination has no parent")?,
    )?;
    if copy {
        let mut input = fs::File::open(&source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)?;
        let copied = io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        if copied != file.size {
            return Err(format!(
                "copied {} bytes for {}, expected {}",
                copied, file.path, file.size
            )
            .into());
        }
    } else {
        fs::hard_link(&source, &destination).map_err(|error| {
            format!(
                "hardlink {} -> {} failed; keep roots on one filesystem or pass --copy: {error}",
                source.display(),
                destination.display()
            )
        })?;
        require_same_file(&source, &destination)?;
    }
    Ok(())
}

fn weight_bytes(manifest: &ArtifactManifest) -> Result<u64, Box<dyn Error>> {
    manifest
        .files
        .iter()
        .filter(|file| file.role == ArtifactFileRole::Weights)
        .try_fold(0_u64, |sum, file| {
            sum.checked_add(file.size)
                .ok_or_else(|| "weight byte count overflow".into())
        })
}

fn weight_count(manifest: &ArtifactManifest) -> usize {
    manifest
        .files
        .iter()
        .filter(|file| file.role == ArtifactFileRole::Weights)
        .count()
}

fn sealed_digest(manifest: &ArtifactManifest) -> String {
    manifest
        .content_digest
        .expect("sealed manifest has a content digest")
        .to_string()
}

fn dependency_from_manifest(
    role: &str,
    manifest: &ArtifactManifest,
) -> Result<ArtifactDependency, Box<dyn Error>> {
    manifest.validate_sealed()?;
    Ok(ArtifactDependency {
        role: ArtifactComponentId::new(role)?,
        bundle: manifest.bundle.clone(),
        profile: manifest.profile.clone(),
        model: manifest.model.clone(),
        model_revision: manifest.model_revision.clone(),
        content_digest: manifest
            .content_digest
            .expect("validated sealed manifest has a content digest"),
    })
}

fn sort_files(files: &mut [ArtifactFile]) {
    files.sort_by(|left, right| left.path.cmp(&right.path));
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<u64, Box<dyn Error>> {
    let bytes = json_bytes(value)?;
    write_bytes_new(path, &bytes)?;
    Ok(u64::try_from(bytes.len())?)
}

fn write_bytes_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<Sha256Digest, Box<dyn Error>> {
    let mut input = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Sha256Digest::from_bytes(hasher.finalize().into()))
}

fn create_temporary_directory(parent: &Path, prefix: &str) -> Result<PathBuf, Box<dyn Error>> {
    for attempt in 0..100_u32 {
        let path = parent.join(format!("{prefix}.{}.{}", std::process::id(), attempt));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(format!(
        "could not create a fresh temporary directory beneath {}",
        parent.display()
    )
    .into())
}

fn require_safe_relative(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe manifest path: {}", path.display()).into());
    }
    Ok(())
}

fn require_real_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("expected a non-symlink directory: {}", path.display()).into());
    }
    Ok(())
}

fn require_regular_path(root: &Path, relative: &Path) -> Result<(), Box<dyn Error>> {
    require_safe_relative(relative)?;
    require_real_directory(root)?;
    let mut current = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(format!("unsafe manifest path: {}", relative.display()).into());
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            return Err(format!("payload path traverses a symlink: {}", current.display()).into());
        }
        if index + 1 == components.len() {
            if !metadata.is_file() {
                return Err(format!("payload is not a regular file: {}", current.display()).into());
            }
        } else if !metadata.is_dir() {
            return Err(format!("payload parent is not a directory: {}", current.display()).into());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn require_same_file(left: &Path, right: &Path) -> Result<(), Box<dyn Error>> {
    use std::os::unix::fs::MetadataExt;

    let left_metadata = fs::symlink_metadata(left)?;
    let right_metadata = fs::symlink_metadata(right)?;
    if left_metadata.dev() != right_metadata.dev() || left_metadata.ino() != right_metadata.ino() {
        return Err(format!(
            "payload is not a hardlink: {} -> {}",
            left.display(),
            right.display()
        )
        .into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_same_file(_left: &Path, _right: &Path) -> Result<(), Box<dyn Error>> {
    Err("hardlink attestation requires a Unix filesystem; pass --copy".into())
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
    };

    use burn_image::{
        ARTIFACT_MANIFEST_SCHEMA_V1, ARTIFACT_TRANSPORT_LAYOUT_PATH,
        ARTIFACT_TRANSPORT_TARGET_PART_BYTES, ArtifactBundleId, ArtifactFile, ArtifactFileRole,
        ArtifactManifest, ArtifactPath, ArtifactProfileId, ArtifactTransportLayout, ModelId,
        NumericFormat, Sha256Digest,
    };

    use super::{
        FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID, Q4_BUNDLES, QWEN_Q4S_COMPONENT_BUNDLE_ID,
        install_transport_layout, transport_stats,
    };

    #[test]
    fn q4_only_release_contains_shared_components_and_every_public_variant_correctness() {
        assert_ne!(
            QWEN_Q4S_COMPONENT_BUNDLE_ID,
            FLUX_VAE_SHARED_COMPONENT_BUNDLE_ID
        );
        assert_eq!(Q4_BUNDLES.len(), 3);
        assert_eq!(
            Q4_BUNDLES
                .iter()
                .map(|bundle| bundle.canonical_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "boogu-image-0.1-turbo-q4s-block-up-to128-f32",
                "boogu-image-0.1-edit-turbo-q4s-block-up-to128-f32",
                "boogu-image-0.1-edit-turbo-1k5-q4s-block-up-to128-f32",
            ])
        );
    }

    fn install_fixture() -> (tempfile::TempDir, ArtifactManifest, Vec<u8>) {
        let directory = tempfile::tempdir().unwrap();
        let logical_path = ArtifactPath::new("objects/fixture.bpk").unwrap();
        let logical_size = usize::try_from(ARTIFACT_TRANSPORT_TARGET_PART_BYTES).unwrap() + 7;
        let bytes = (0..logical_size)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        fs::create_dir_all(directory.path().join("objects")).unwrap();
        fs::create_dir_all(directory.path().join("metadata")).unwrap();
        fs::write(directory.path().join(logical_path.as_str()), &bytes).unwrap();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "target_max_shard_bytes".into(),
            (256_u64 * 1024 * 1024).to_string(),
        );
        let mut manifest = ArtifactManifest {
            schema_version: ARTIFACT_MANIFEST_SCHEMA_V1,
            bundle: ArtifactBundleId::new("transport-generator-test").unwrap(),
            profile: ArtifactProfileId::new("f16").unwrap(),
            model: ModelId::new("tests/transport-generator").unwrap(),
            model_revision: "immutable-test-revision".into(),
            numeric_format: NumericFormat::F16,
            components: Vec::new(),
            files: vec![ArtifactFile {
                path: logical_path,
                size: bytes.len() as u64,
                sha256: Sha256Digest::calculate(&bytes),
                role: ArtifactFileRole::Weights,
                component: None,
                shard: None,
            }],
            dependencies: Vec::new(),
            metadata,
            content_digest: None,
        };
        manifest.seal().unwrap();
        install_transport_layout(directory.path(), &mut manifest).unwrap();
        (directory, manifest, bytes)
    }

    #[test]
    fn transport_parts_are_bounded_content_addressed_and_deterministic_correctness() {
        let (directory, manifest, original) = install_fixture();
        assert!(!directory.path().join("objects/fixture.bpk").exists());
        let layout_bytes = fs::read(directory.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH)).unwrap();
        let layout = ArtifactTransportLayout::parse_and_validate(&manifest, &layout_bytes).unwrap();
        let object = &layout.objects()[0];
        assert_eq!(object.parts.len(), 2);
        assert_eq!(object.parts[0].size, ARTIFACT_TRANSPORT_TARGET_PART_BYTES);
        assert_eq!(object.parts[1].size, 7);
        let reconstructed = object
            .parts
            .iter()
            .flat_map(|part| fs::read(directory.path().join(part.path.as_str())).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(reconstructed, original);

        let stats = transport_stats(directory.path(), &manifest).unwrap();
        assert_eq!(stats.objects, 1);
        assert_eq!(stats.parts, 2);
        assert_eq!(stats.bytes, original.len() as u64);
        assert_eq!(
            stats.largest_part_bytes,
            ARTIFACT_TRANSPORT_TARGET_PART_BYTES
        );

        let (second_directory, second_manifest, _) = install_fixture();
        assert_eq!(manifest.content_digest, second_manifest.content_digest);
        assert_eq!(
            layout_bytes,
            fs::read(second_directory.path().join(ARTIFACT_TRANSPORT_LAYOUT_PATH)).unwrap()
        );
    }

    #[test]
    fn generator_plan_rejects_direct_metadata_above_physical_ceiling_correctness() {
        let (directory, mut manifest, _original) = install_fixture();
        manifest.files.push(ArtifactFile {
            path: ArtifactPath::new("metadata/oversized.bin").unwrap(),
            size: burn_image::ARTIFACT_TRANSPORT_MAX_PART_BYTES + 1,
            sha256: Sha256Digest::calculate(b"oversized-fixture"),
            role: ArtifactFileRole::Other,
            component: None,
            shard: None,
        });
        manifest.seal().unwrap();

        let error = transport_stats(directory.path(), &manifest).unwrap_err();
        assert!(error.to_string().contains("physical CDN object cap"));
    }
}
