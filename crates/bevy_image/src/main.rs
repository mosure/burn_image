#[cfg(not(feature = "boogu-native"))]
fn main() {
    bevy_burn_image::app::build_app().run();
}

#[cfg(feature = "boogu-native")]
fn main() {
    use bevy_burn_image::{
        BooguAdapterSettings, NativeBooguFactory, NativeBooguResidencyPolicy,
        default_native_boogu_model_base_url, run_boogu_cli,
    };
    use burn_boogu::{BooguVariant, artifacts::BooguStorageProfile};
    use burn_image::ArtifactSource;
    use clap::{Parser, ValueEnum};

    #[derive(Clone, Copy, Debug, ValueEnum)]
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

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum ProfileArg {
        F16,
        F16QwenVisionF32,
        Q8sBlock32F32,
        Q8sBlock32F32QwenVisionF32,
    }

    impl From<ProfileArg> for BooguStorageProfile {
        fn from(value: ProfileArg) -> Self {
            match value {
                ProfileArg::F16 => Self::F16,
                ProfileArg::F16QwenVisionF32 => Self::F16QwenVisionF32,
                ProfileArg::Q8sBlock32F32 => Self::Q8sBlock32F32,
                ProfileArg::Q8sBlock32F32QwenVisionF32 => Self::Q8sBlock32F32QwenVisionF32,
            }
        }
    }

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum ResidencyArg {
        NativeHighVram,
        NativeLayerStreamed,
    }

    impl From<ResidencyArg> for NativeBooguResidencyPolicy {
        fn from(value: ResidencyArg) -> Self {
            match value {
                ResidencyArg::NativeHighVram => Self::HighVram,
                ResidencyArg::NativeLayerStreamed => Self::LayerStreamed,
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
        /// Numeric profile represented by the bundle.
        #[arg(long, value_enum, default_value = "f16-qwen-vision-f32")]
        profile: ProfileArg,
        /// Native residency. High-VRAM retains Qwen and the denoiser; layer-streamed rereads them.
        #[arg(long, value_enum, default_value = "native-high-vram")]
        residency: ResidencyArg,
    }

    let args = Args::parse();
    let variant = args.variant.into();
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
    run_boogu_cli(
        settings,
        NativeBooguFactory::with_residency(variant, residency),
    );
}
