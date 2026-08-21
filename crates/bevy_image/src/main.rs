#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ProfileArg {
    /// Production default: packed Q4S for every public release.
    #[value(name = "production")]
    Production,
    #[value(name = "f16-qwen-vision-f32")]
    F16QwenVisionF32,
    #[value(name = "q4s")]
    Q4sBlockUpTo128F32,
}

#[cfg(feature = "boogu-native")]
impl ProfileArg {
    fn resolve(
        self,
        variant: burn_boogu::BooguVariant,
    ) -> burn_boogu::artifacts::BooguStorageProfile {
        match self {
            ProfileArg::Production => bevy_burn_image::default_boogu_storage_profile(variant),
            ProfileArg::F16QwenVisionF32 => {
                burn_boogu::artifacts::BooguStorageProfile::F16QwenVisionF32
            }
            ProfileArg::Q4sBlockUpTo128F32 => {
                burn_boogu::artifacts::BooguStorageProfile::Q4sBlockUpTo128F32
            }
        }
    }
}

#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ResidencyArg {
    NativeHighVram,
    LowVram,
}

#[cfg(feature = "boogu-native")]
impl From<ResidencyArg> for bevy_burn_image::NativeBooguResidencyPolicy {
    fn from(value: ResidencyArg) -> Self {
        match value {
            ResidencyArg::NativeHighVram => Self::HighVram,
            ResidencyArg::LowVram => Self::LowVram,
        }
    }
}

#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum AutotuneArg {
    Balanced,
    Full,
}

#[cfg(feature = "boogu-native")]
impl From<AutotuneArg> for burn_boogu::NativeAutotunePolicy {
    fn from(value: AutotuneArg) -> Self {
        match value {
            AutotuneArg::Balanced => Self::Balanced,
            AutotuneArg::Full => Self::Full,
        }
    }
}

#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum VariantArg {
    Turbo,
    EditTurbo,
    #[value(name = "edit-turbo-1k5")]
    EditTurbo1k5,
}

#[cfg(feature = "boogu-native")]
impl From<VariantArg> for burn_boogu::BooguVariant {
    fn from(value: VariantArg) -> Self {
        match value {
            VariantArg::Turbo => Self::Image01Turbo,
            VariantArg::EditTurbo => Self::Image01EditTurbo,
            VariantArg::EditTurbo1k5 => Self::Image01EditTurbo1k5,
        }
    }
}

#[cfg(feature = "boogu-native")]
#[derive(Debug, clap::Parser)]
#[command(
    about = "Run the Bevy frontend with a sealed Boogu bundle",
    after_help = "UNATTENDED EXAMPLES:\n  Generate:\n    bevy_image --variant turbo --prompt 'a blue ceramic bird' --output result.png\n\n  Edit:\n    bevy_image --variant edit-turbo --prompt 'make the bird red' --source input.jpg --output result.png\n\nSupplying --output enables unattended mode: the GPU runtime is initialized, one ordinary image job is submitted, the PNG and JSON report are written, and the process exits without UI interaction."
)]
struct Args {
    /// Local sealed bundle override. Omit to use the verified Aberration CDN cache.
    #[arg(long)]
    artifacts: Option<std::path::PathBuf>,
    /// Initially selected immutable Boogu release. The interactive canonical UI can switch to
    /// Generate, Edit 1K, or Edit 1.5K without restarting.
    #[arg(long, value_enum, default_value = "turbo")]
    variant: VariantArg,
    /// Storage profile represented by the bundle. `production` uses packed Q4S for every variant.
    #[arg(long, value_enum, default_value = "production")]
    profile: ProfileArg,
    /// Native residency. `low-vram` streams Qwen/VAE per request while retaining the denoiser below a fail-closed 32-GB device plan.
    #[arg(long, value_enum, default_value = "native-high-vram")]
    residency: ResidencyArg,
    /// CubeCL kernel tuning policy. Available only in a `native-autotune` build; ordinary builds
    /// use static kernels and avoid first-use tuning pauses.
    #[arg(long, value_enum)]
    autotune: Option<AutotuneArg>,
    /// Execute one request without UI interaction, save this PNG, write a JSON timing report,
    /// and exit.
    #[arg(
        long,
        value_name = "PNG",
        requires = "prompt",
        conflicts_with = "qualification_output_dir"
    )]
    output: Option<std::path::PathBuf>,
    /// Prompt or edit instruction for --output.
    #[arg(long, requires = "output")]
    prompt: Option<String>,
    /// Reference image for an Edit variant; required for Edit and forbidden for Turbo.
    #[arg(long, value_name = "IMAGE", requires = "output")]
    source: Option<std::path::PathBuf>,
    /// Output width for --output; defaults to 1024 or the 1.5K release's 1536.
    #[arg(long, requires = "output")]
    width: Option<u32>,
    /// Output height for --output; defaults to 1024 or the 1.5K release's 1536.
    #[arg(long, requires = "output")]
    height: Option<u32>,
    /// Deterministic request seed for --output (default: 0).
    #[arg(long, requires = "output")]
    seed: Option<u64>,
    /// JSON timing/provenance report for --output; defaults beside the PNG.
    #[arg(long, value_name = "JSON", requires = "output")]
    report: Option<std::path::PathBuf>,
    /// Fail an unattended run after this many seconds (default: 7200).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..), requires = "output")]
    timeout_seconds: Option<u64>,
    /// Keep the Bevy window visible during an unattended run. It is hidden by default.
    #[arg(long, requires = "output")]
    show_window: bool,
    /// Run one exact released native/browser output-comparison candidate through the ordinary
    /// Bevy frontend, write its production Save-PNG output and report here, then exit.
    #[arg(long, value_name = "DIRECTORY")]
    qualification_output_dir: Option<std::path::PathBuf>,
    /// Exact qualification prompt or edit instruction.
    #[arg(long, requires = "qualification_output_dir")]
    qualification_prompt: Option<String>,
    /// Exact qualification seed, including zero.
    #[arg(long, requires = "qualification_output_dir")]
    qualification_seed: Option<u64>,
    /// Exact released output width.
    #[arg(long, requires = "qualification_output_dir")]
    qualification_width: Option<u32>,
    /// Exact released output height.
    #[arg(long, requires = "qualification_output_dir")]
    qualification_height: Option<u32>,
    /// Original bounded source image for Edit-Turbo; required for edit and forbidden for Turbo.
    #[arg(long, requires = "qualification_output_dir")]
    qualification_source: Option<std::path::PathBuf>,
}

#[cfg(feature = "boogu-native")]
fn validate_autotune_feature_request(args: &Args) -> Result<(), &'static str> {
    if cfg!(feature = "native-autotune") {
        return Ok(());
    }
    if args.autotune.is_some() {
        return Err("--autotune requires building bevy_image with --features native-autotune");
    }
    if args.qualification_output_dir.is_some() {
        return Err(
            "--qualification-output-dir requires building bevy_image with --features native-autotune",
        );
    }
    Ok(())
}

#[cfg(feature = "boogu-native")]
fn main() -> bevy::app::AppExit {
    use bevy_burn_image::{
        BooguAdapterSettings, NativeAutomatedRun, NativeAutomatedRunPlugin, NativeBooguFactory,
        NativeOutputQualification, NativeOutputQualificationPlugin,
        default_native_boogu_model_base_url, prepare_native_output_request,
        verify_native_output_artifacts,
    };
    use burn_boogu::BooguVariant;
    use burn_image::ArtifactSource;
    use clap::Parser;
    let args = Args::parse();
    if let Err(error) = validate_autotune_feature_request(&args) {
        clap::Error::raw(clap::error::ErrorKind::ValueValidation, error).exit();
    }
    let variant: BooguVariant = args.variant.into();
    let interactive_canonical_model_switching = args.output.is_none()
        && args.qualification_output_dir.is_none()
        && args.artifacts.is_none()
        && std::env::var_os("BURN_IMAGE_MODEL_BASE_URL").is_none()
        && args.profile == ProfileArg::Production;
    let autotune = args.autotune.map(Into::into).unwrap_or_else(|| {
        if args.qualification_output_dir.is_some() {
            burn_boogu::NativeAutotunePolicy::Full
        } else {
            burn_boogu::NativeAutotunePolicy::Balanced
        }
    });
    if args.qualification_output_dir.is_some() && autotune != burn_boogu::NativeAutotunePolicy::Full
    {
        clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            "--qualification-output-dir requires --autotune full (or omit --autotune to select it automatically)",
        )
        .exit();
    }
    let automated_run = args.output.as_ref().map(|output_path| {
        if !output_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                "--output must end in .png",
            )
            .exit();
        }
        let edge = match variant {
            BooguVariant::Image01EditTurbo1k5 => burn_boogu::BOOGU_1K5_DEFAULT_EDGE,
            BooguVariant::Image01Turbo | BooguVariant::Image01EditTurbo => {
                burn_boogu::BOOGU_DEFAULT_EDGE
            }
        };
        let prompt = args
            .prompt
            .clone()
            .expect("clap requires --prompt with --output");
        let (request, request_identity) = prepare_native_output_request(
            variant,
            prompt,
            args.seed.unwrap_or(0),
            args.width.unwrap_or(edge),
            args.height.unwrap_or(edge),
            args.source.clone(),
        )
        .unwrap_or_else(|error| {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                format!("native automated request validation failed: {error}"),
            )
            .exit()
        });
        let report_path = args
            .report
            .clone()
            .unwrap_or_else(|| output_path.with_extension("report.json"));
        if report_path == *output_path {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                "--report must differ from --output",
            )
            .exit();
        }
        NativeAutomatedRun {
            variant,
            request,
            request_identity,
            output_path: output_path.clone(),
            report_path,
            timeout: std::time::Duration::from_secs(args.timeout_seconds.unwrap_or(7_200)),
            show_window: args.show_window,
        }
    });
    let qualification = args.qualification_output_dir.as_ref().map(|output_directory| {
        if args.profile != ProfileArg::Production || args.residency != ResidencyArg::LowVram {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                "--qualification-output-dir requires --profile production --residency low-vram",
            )
            .exit();
        }
        let artifact_root = args.artifacts.as_ref().unwrap_or_else(|| {
            clap::Error::raw(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--qualification-output-dir requires an explicit local --artifacts canonical modular root",
            )
            .exit()
        });
        macro_rules! required {
            ($name:literal, $value:expr) => {
                $value.unwrap_or_else(|| {
                clap::Error::raw(
                    clap::error::ErrorKind::MissingRequiredArgument,
                        concat!(
                            "--qualification-output-dir requires --qualification-",
                            $name
                        ),
                )
                .exit()
                })
            };
        }
        let prompt = required!("prompt", args.qualification_prompt.clone());
        let seed = required!("seed", args.qualification_seed);
        let width = required!("width", args.qualification_width);
        let height = required!("height", args.qualification_height);
        let (request, request_identity) = prepare_native_output_request(
            variant,
            prompt,
            seed,
            width,
            height,
            args.qualification_source.clone(),
        )
        .unwrap_or_else(|error| {
            clap::Error::raw(
                clap::error::ErrorKind::ValueValidation,
                format!("native output request qualification failed: {error}"),
            )
            .exit()
        });
        eprintln!("authenticating the complete canonical modular release closure");
        let artifacts = verify_native_output_artifacts(artifact_root, variant).unwrap_or_else(
            |error| {
                clap::Error::raw(
                    clap::error::ErrorKind::ValueValidation,
                    format!("native output artifact qualification failed: {error}"),
                )
                .exit()
            },
        );
        NativeOutputQualification {
            variant,
            request,
            request_identity,
            output_directory: output_directory.clone(),
            artifacts,
        }
    });
    let profile =
        if args.qualification_output_dir.is_some() && args.profile == ProfileArg::Production {
            burn_boogu::artifacts::BooguStorageProfile::F16QwenVisionF32
        } else {
            args.profile.resolve(variant)
        };
    let residency = args.residency.into();
    #[cfg(feature = "native-autotune")]
    burn_boogu::configure_native_autotune(autotune);
    let artifact_source = match args.artifacts {
        Some(root) => ArtifactSource::LocalDirectory { root },
        None => ArtifactSource::Remote {
            base_url: default_native_boogu_model_base_url(variant, profile).unwrap_or_else(
                |error| clap::Error::raw(clap::error::ErrorKind::ValueValidation, error).exit(),
            ),
        },
    };
    let mut settings = BooguAdapterSettings::production(variant, artifact_source);
    settings.storage_profile = profile;
    let factory = if interactive_canonical_model_switching {
        NativeBooguFactory::with_canonical_model_switching(variant, residency, autotune)
    } else {
        NativeBooguFactory::with_residency_and_autotune(variant, residency, autotune)
    };
    let mut app = bevy_burn_image::app::build_boogu_app(settings, factory);
    if let Some(qualification) = qualification {
        app.add_plugins(NativeOutputQualificationPlugin::new(qualification));
    }
    if let Some(automated_run) = automated_run {
        app.add_plugins(NativeAutomatedRunPlugin::new(automated_run));
    }
    app.run()
}

#[cfg(all(test, feature = "boogu-native"))]
mod tests {
    use super::{
        Args, AutotuneArg, ProfileArg, ResidencyArg, VariantArg, validate_autotune_feature_request,
    };
    use bevy_burn_image::NativeBooguResidencyPolicy;
    use burn_boogu::artifacts::BooguStorageProfile;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn native_profile_resolves_current_release_profiles_correctness() {
        let production = <ProfileArg as clap::ValueEnum>::from_str("production", false).unwrap();
        let mixed_f16 =
            <ProfileArg as clap::ValueEnum>::from_str("f16-qwen-vision-f32", false).unwrap();
        assert_eq!(
            production.resolve(burn_boogu::BooguVariant::Image01Turbo),
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        assert_eq!(
            production.resolve(burn_boogu::BooguVariant::Image01EditTurbo),
            BooguStorageProfile::Q4sBlockUpTo128F32
        );
        assert_eq!(
            mixed_f16.resolve(burn_boogu::BooguVariant::Image01Turbo),
            BooguStorageProfile::F16QwenVisionF32
        );
    }

    #[test]
    fn native_cli_accepts_supported_low_vram_residency_correctness() {
        let low_vram = <ResidencyArg as clap::ValueEnum>::from_str("low-vram", false).unwrap();
        assert_eq!(
            NativeBooguResidencyPolicy::from(low_vram),
            NativeBooguResidencyPolicy::LowVram
        );
    }

    #[test]
    fn native_cli_defaults_interactive_ui_to_generate_turbo_correctness() {
        let args = Args::try_parse_from(["bevy_image"]).unwrap();
        assert_eq!(args.variant, VariantArg::Turbo);
        assert!(args.output.is_none());
    }

    #[test]
    fn default_install_builds_only_the_native_wgpu_application_correctness() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("default-run = \"bevy_image\""));
        assert!(manifest.contains("default = [\"boogu-native\"]"));
        let bins = manifest
            .split_once("[[bin]]")
            .expect("bevy_image manifest must declare its application binary")
            .1;
        let (application, qualification) = bins
            .split_once("[[bin]]")
            .expect("bevy_image manifest must isolate its qualification helper");
        assert!(application.contains("name = \"bevy_image\""));
        assert!(application.contains("required-features = [\"boogu-native\"]"));
        assert!(qualification.contains("name = \"burn-image-output-quality\""));
        assert!(qualification.contains("required-features = [\"output-quality\"]"));
    }

    #[test]
    fn native_cli_maps_interactive_and_qualification_autotune_policies_correctness() {
        let balanced = <AutotuneArg as clap::ValueEnum>::from_str("balanced", false).unwrap();
        let full = <AutotuneArg as clap::ValueEnum>::from_str("full", false).unwrap();
        assert_eq!(
            burn_boogu::NativeAutotunePolicy::from(balanced),
            burn_boogu::NativeAutotunePolicy::Balanced
        );
        assert_eq!(
            burn_boogu::NativeAutotunePolicy::from(full),
            burn_boogu::NativeAutotunePolicy::Full
        );
    }

    #[test]
    fn native_cli_autotune_option_tracks_the_build_feature_correctness() {
        let ordinary = Args::try_parse_from(["bevy_image"]).unwrap();
        assert!(validate_autotune_feature_request(&ordinary).is_ok());

        let requested = Args::try_parse_from(["bevy_image", "--autotune", "balanced"]).unwrap();
        assert_eq!(
            validate_autotune_feature_request(&requested).is_ok(),
            cfg!(feature = "native-autotune")
        );
    }

    #[test]
    fn native_cli_parses_unattended_generate_contract_correctness() {
        let args = Args::try_parse_from([
            "bevy_image",
            "--variant",
            "turbo",
            "--prompt",
            "a blue ceramic bird",
            "--output",
            "result.PNG",
            "--report",
            "result.json",
            "--width",
            "1024",
            "--height",
            "1024",
            "--seed",
            "17",
            "--timeout-seconds",
            "900",
        ])
        .unwrap();
        assert_eq!(args.variant, VariantArg::Turbo);
        assert_eq!(args.prompt.as_deref(), Some("a blue ceramic bird"));
        assert_eq!(args.output, Some(PathBuf::from("result.PNG")));
        assert_eq!(args.report, Some(PathBuf::from("result.json")));
        assert_eq!((args.width, args.height), (Some(1024), Some(1024)));
        assert_eq!(args.seed, Some(17));
        assert_eq!(args.timeout_seconds, Some(900));
        assert!(!args.show_window);
    }

    #[test]
    fn native_cli_parses_unattended_edit_source_and_visible_debug_window_correctness() {
        let args = Args::try_parse_from([
            "bevy_image",
            "--variant",
            "edit-turbo",
            "--prompt",
            "make the bird red",
            "--source",
            "input.jpg",
            "--output",
            "result.png",
            "--show-window",
        ])
        .unwrap();
        assert_eq!(args.variant, VariantArg::EditTurbo);
        assert_eq!(args.source, Some(PathBuf::from("input.jpg")));
        assert!(args.show_window);
    }

    #[test]
    fn native_cli_rejects_incomplete_unattended_invocation_correctness() {
        let missing_prompt =
            Args::try_parse_from(["bevy_image", "--variant", "turbo", "--output", "result.png"])
                .unwrap_err();
        assert_eq!(
            missing_prompt.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let orphan_prompt =
            Args::try_parse_from(["bevy_image", "--variant", "turbo", "--prompt", "a bird"])
                .unwrap_err();
        assert_eq!(
            orphan_prompt.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }
}
