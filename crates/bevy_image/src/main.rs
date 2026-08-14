#[cfg(not(feature = "boogu-native"))]
fn main() {
    bevy_burn_image::app::build_app().run();
}

#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ProfileArg {
    F16,
    #[value(name = "production", alias = "f16-qwen-vision-f32")]
    Production,
    Q8sBlock32F32,
    Q8sBlock32F32QwenVisionF32,
}

#[cfg(feature = "boogu-native")]
impl From<ProfileArg> for burn_boogu::artifacts::BooguStorageProfile {
    fn from(value: ProfileArg) -> Self {
        match value {
            ProfileArg::F16 => Self::F16,
            ProfileArg::Production => Self::F16QwenVisionF32,
            ProfileArg::Q8sBlock32F32 => Self::Q8sBlock32F32,
            ProfileArg::Q8sBlock32F32QwenVisionF32 => Self::Q8sBlock32F32QwenVisionF32,
        }
    }
}

#[cfg(feature = "boogu-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ResidencyArg {
    NativeHighVram,
    LowVram,
    DiagnosticLayerStreamed,
}

#[cfg(feature = "boogu-native")]
impl From<ResidencyArg> for bevy_burn_image::NativeBooguResidencyPolicy {
    fn from(value: ResidencyArg) -> Self {
        match value {
            ResidencyArg::NativeHighVram => Self::HighVram,
            ResidencyArg::LowVram => Self::LowVram,
            ResidencyArg::DiagnosticLayerStreamed => Self::LayerStreamed,
        }
    }
}

#[cfg(feature = "boogu-native")]
fn main() -> bevy::app::AppExit {
    use bevy_burn_image::{
        BooguAdapterSettings, NativeBooguFactory, NativeOutputQualification,
        NativeOutputQualificationPlugin, default_native_boogu_model_base_url,
        prepare_native_output_qualification_request, verify_native_output_artifacts,
    };
    use burn_boogu::BooguVariant;
    use burn_image::ArtifactSource;
    use clap::{Parser, ValueEnum};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
    enum VariantArg {
        Turbo,
        EditTurbo,
        #[value(name = "edit-turbo-1k5")]
        EditTurbo1k5,
    }

    impl From<VariantArg> for BooguVariant {
        fn from(value: VariantArg) -> Self {
            match value {
                VariantArg::Turbo => Self::Image01Turbo,
                VariantArg::EditTurbo => Self::Image01EditTurbo,
                VariantArg::EditTurbo1k5 => Self::Image01EditTurbo1k5,
            }
        }
    }

    #[derive(Debug, Parser)]
    #[command(about = "Run the Bevy frontend with a sealed Boogu bundle")]
    struct Args {
        /// Local sealed bundle override. Omit to use the verified Aberration CDN cache.
        #[arg(long)]
        artifacts: Option<std::path::PathBuf>,
        /// Immutable Boogu release represented by the bundle.
        #[arg(long, value_enum)]
        variant: VariantArg,
        /// Storage profile represented by the bundle. `production` uses mixed F16 with F32 Qwen vision.
        #[arg(long, value_enum, default_value = "production")]
        profile: ProfileArg,
        /// Native residency. `low-vram` streams Qwen/VAE per request while retaining the denoiser below a fail-closed 32-GB device plan.
        #[arg(long, value_enum, default_value = "native-high-vram")]
        residency: ResidencyArg,
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

    let args = Args::parse();
    let variant: BooguVariant = args.variant.into();
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
        let (request, request_identity) = prepare_native_output_qualification_request(
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
    let profile = args.profile.into();
    let residency = args.residency.into();
    if NativeBooguFactory::requires_full_autotune(variant, residency, profile) {
        burn_boogu::configure_native_full_autotune();
    }
    let artifact_source = match args.artifacts {
        Some(root) => ArtifactSource::LocalDirectory { root },
        None => ArtifactSource::Remote {
            base_url: default_native_boogu_model_base_url(variant, profile).unwrap_or_else(
                |error| clap::Error::raw(clap::error::ErrorKind::ValueValidation, error).exit(),
            ),
        },
    };
    let mut settings = BooguAdapterSettings::verified_default(artifact_source);
    settings.storage_profile = profile;
    let mut app = bevy_burn_image::app::build_boogu_app(
        settings,
        NativeBooguFactory::with_residency(variant, residency),
    );
    if let Some(qualification) = qualification {
        app.add_plugins(NativeOutputQualificationPlugin::new(qualification));
    }
    app.run()
}

#[cfg(all(test, feature = "boogu-native"))]
mod tests {
    use super::{ProfileArg, ResidencyArg};
    use bevy_burn_image::NativeBooguResidencyPolicy;
    use burn_boogu::artifacts::BooguStorageProfile;

    #[test]
    fn native_profile_prefers_production_and_accepts_legacy_alias_correctness() {
        let production = <ProfileArg as clap::ValueEnum>::from_str("production", false).unwrap();
        let legacy =
            <ProfileArg as clap::ValueEnum>::from_str("f16-qwen-vision-f32", false).unwrap();
        assert_eq!(
            BooguStorageProfile::from(production),
            BooguStorageProfile::F16QwenVisionF32
        );
        assert_eq!(
            BooguStorageProfile::from(legacy),
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
}
