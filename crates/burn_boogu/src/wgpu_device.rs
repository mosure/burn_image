//! Native hardware-adapter selection for standalone WGPU tools.

use crate::BooguError;
#[cfg(all(feature = "autotune", not(target_arch = "wasm32")))]
use crate::NativeAutotunePolicy;

/// Configure CubeCL's native autotune policy before any WGPU device is initialized.
///
/// Interactive applications should use [`NativeAutotunePolicy::Balanced`], which buckets nearby
/// tensor shapes so a new prompt or reference does not trigger an exact-shape tuning storm. Full
/// autotune remains available for synchronized performance qualification and long-running,
/// fixed-shape throughput workloads.
#[cfg(all(feature = "autotune", not(target_arch = "wasm32")))]
pub fn configure_native_autotune(policy: NativeAutotunePolicy) {
    use burn_cubecl::cubecl::config::{
        CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel,
    };

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.autotune.level = match policy {
        NativeAutotunePolicy::Balanced => AutotuneLevel::Balanced,
        NativeAutotunePolicy::Full => AutotuneLevel::Full,
    };
    CubeClRuntimeConfig::set(config);
}

/// Select CubeCL's exhaustive autotune search before any native WGPU device is initialized.
///
/// Qualified native high-VRAM mixed-F16 performance gates are calibrated against the full
/// candidate search. Call this once, on the main thread, before Bevy/Burn creates or imports a
/// WGPU device. CubeCL deliberately rejects configuration changes after its global runtime
/// configuration has been observed.
#[cfg(all(feature = "autotune", not(target_arch = "wasm32")))]
pub fn configure_native_full_autotune() {
    configure_native_autotune(NativeAutotunePolicy::Full);
}

/// Require that the process selected the requested CubeCL autotune policy.
#[cfg(all(feature = "autotune", not(target_arch = "wasm32")))]
pub fn require_native_autotune_configured(
    expected: NativeAutotunePolicy,
) -> Result<(), BooguError> {
    use burn_cubecl::cubecl::config::{
        CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel,
    };

    let config = CubeClRuntimeConfig::get();
    let matches = matches!(
        (&expected, &config.autotune.level),
        (NativeAutotunePolicy::Balanced, AutotuneLevel::Balanced)
            | (NativeAutotunePolicy::Full, AutotuneLevel::Full)
    );
    if matches {
        Ok(())
    } else {
        Err(BooguError::InvalidConfig(format!(
            "native execution requires CubeCL {} autotune configured before creating a WGPU device",
            match expected {
                NativeAutotunePolicy::Balanced => "balanced",
                NativeAutotunePolicy::Full => "full",
            }
        )))
    }
}

/// Require that the process selected CubeCL's exhaustive autotune search.
///
/// This check intentionally observes the global runtime configuration. Call
/// [`configure_native_full_autotune`] before any device is created; an embedding that omits that
/// setup fails closed instead of reporting a calibrated production policy.
#[cfg(all(feature = "autotune", not(target_arch = "wasm32")))]
pub fn require_native_full_autotune_configured() -> Result<(), BooguError> {
    require_native_autotune_configured(NativeAutotunePolicy::Full).map_err(|_| {
        BooguError::InvalidConfig(
            "the qualified native high-VRAM mixed-F16 policy requires CubeCL full autotune; call \
             configure_native_full_autotune() before creating a WGPU device"
                .into(),
        )
    })
}

/// Initialize Burn WGPU on a native hardware adapter and reject CPU/software adapters.
///
/// Burn's `DefaultDevice` can be redirected to `Cpu` through
/// `CUBECL_WGPU_DEFAULT_DEVICE`. Initializing and inspecting the exact selected setup here keeps
/// release runners and parity tools from silently reporting a CPU-backed execution as WGPU.
#[cfg(not(target_arch = "wasm32"))]
pub fn require_native_wgpu_device() -> Result<burn_wgpu::WgpuDevice, BooguError> {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    let setup = burn_wgpu::init_setup::<burn_wgpu::graphics::AutoGraphicsApi>(
        &device,
        burn_wgpu::RuntimeOptions::default(),
    );
    let info = setup.adapter.get_info();
    validate_native_adapter_type(info.device_type).map_err(|reason| {
        BooguError::InvalidConfig(format!(
            "standalone WGPU requires a hardware adapter, selected {:?} ({:?}): {reason}",
            info.name, info.device_type
        ))
    })?;
    Ok(device)
}

/// Standalone native WGPU runners are not available in a Wasm binary.
#[cfg(target_arch = "wasm32")]
pub fn require_native_wgpu_device() -> Result<burn_wgpu::WgpuDevice, BooguError> {
    Err(BooguError::InvalidConfig(
        "standalone native WGPU device selection is unavailable on wasm32".into(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_native_adapter_type(device_type: wgpu::DeviceType) -> Result<(), &'static str> {
    match device_type {
        wgpu::DeviceType::DiscreteGpu
        | wgpu::DeviceType::IntegratedGpu
        | wgpu::DeviceType::VirtualGpu => Ok(()),
        wgpu::DeviceType::Cpu => Err("the selected adapter is explicitly CPU-backed"),
        wgpu::DeviceType::Other => Err("the selected native adapter is not hardware-attestable"),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::validate_native_adapter_type;

    #[test]
    fn native_adapter_policy_rejects_cpu_and_unattested_other_correctness() {
        assert!(validate_native_adapter_type(wgpu::DeviceType::DiscreteGpu).is_ok());
        assert!(validate_native_adapter_type(wgpu::DeviceType::IntegratedGpu).is_ok());
        assert!(validate_native_adapter_type(wgpu::DeviceType::VirtualGpu).is_ok());
        assert!(validate_native_adapter_type(wgpu::DeviceType::Cpu).is_err());
        assert!(validate_native_adapter_type(wgpu::DeviceType::Other).is_err());
    }
}
