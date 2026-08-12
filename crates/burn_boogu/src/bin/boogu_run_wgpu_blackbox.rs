//! Production native WGPU runner requiring parity-gated padded Cubek blackbox FlashAttention.

#[path = "support/boogu_run.rs"]
mod boogu_run;

type Backend = burn_boogu::NativeWgpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    burn_boogu::configure_native_full_autotune();
    boogu_run::run::<Backend, burn_boogu::NativePaddedBlackboxDenoiser>(
        "burn-wgpu-native-padded-blackbox-full-autotune",
        || Ok(burn_boogu::require_native_wgpu_device()?),
        burn_boogu::NativePaddedBlackboxDenoiser::new,
        "required-padded-blackbox-p4-kv1-q1-full-autotune",
    )
}
