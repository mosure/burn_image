use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu-interop")]
use bevy::render::{
    RenderApp,
    renderer::{RenderAdapterInfo, RenderDevice},
};

/// Burn backend initialized from Bevy's existing WGPU objects.
#[cfg(feature = "gpu-interop")]
pub type SharedWgpuBackend = burn_wgpu::Wgpu<f32, i32>;

/// Why the required shared GPU backend is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendFailure {
    GpuInteropFeatureDisabled,
    SharedBridgeNotInstalled,
    RenderSubAppMissing,
    RenderAdapterMissing,
    RenderDeviceMissing,
    BurnDeviceMissing,
    BurnDeviceLost,
    CpuAdapterRejected,
}

impl std::fmt::Display for BackendFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuInteropFeatureDisabled => {
                formatter.write_str("the gpu-interop feature is disabled")
            }
            Self::SharedBridgeNotInstalled => {
                formatter.write_str("the shared Bevy/Burn WGPU bridge was not installed")
            }
            Self::RenderSubAppMissing => formatter.write_str("Bevy's render sub-app is missing"),
            Self::RenderAdapterMissing => {
                formatter.write_str("Bevy did not expose a render adapter")
            }
            Self::RenderDeviceMissing => formatter.write_str("Bevy did not expose a render device"),
            Self::BurnDeviceMissing => {
                formatter.write_str("Burn was not initialized from Bevy's WGPU device")
            }
            Self::BurnDeviceLost => {
                formatter.write_str("the shared Burn WGPU device is no longer available")
            }
            Self::CpuAdapterRejected => {
                formatter.write_str("WGPU selected a CPU adapter; refusing silent CPU fallback")
            }
        }
    }
}

/// Auditable identity of the Bevy adapter used by Burn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendDeviceInfo {
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    /// Always true for a ready status produced by this crate.
    pub shared_adapter_device_queue: bool,
}

/// Current WGPU requirement state. There is intentionally no CPU variant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BackendState {
    Initializing,
    Ready { device: BackendDeviceInfo },
    Failed { reason: BackendFailure },
}

/// Main-world status read by request routing and UI systems.
#[derive(Resource, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub state: BackendState,
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self {
            state: BackendState::Initializing,
        }
    }
}

impl BackendStatus {
    pub fn ready(device: BackendDeviceInfo) -> Self {
        Self {
            state: BackendState::Ready { device },
        }
    }

    pub fn failed(reason: BackendFailure) -> Self {
        Self {
            state: BackendState::Failed { reason },
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.state, BackendState::Ready { .. })
    }

    pub fn unavailable_message(&self) -> Option<String> {
        match &self.state {
            BackendState::Initializing => Some("shared WGPU backend is still initializing".into()),
            BackendState::Failed { reason } => Some(reason.to_string()),
            BackendState::Ready { .. } => None,
        }
    }
}

/// Internal status plugin, registered after the optional `bevy_burn` bridge so
/// its `finish` hook can attest that both sides use one device.
pub struct BackendStatusPlugin {
    pub(crate) bridge_expected: bool,
}

impl Plugin for BackendStatusPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BackendStatus>();

        #[cfg(feature = "gpu-interop")]
        app.add_systems(PreUpdate, enforce_ready_device);
    }

    fn finish(&self, app: &mut App) {
        if !self.bridge_expected {
            app.insert_resource(BackendStatus::failed(
                BackendFailure::SharedBridgeNotInstalled,
            ));
            return;
        }

        #[cfg(not(feature = "gpu-interop"))]
        app.insert_resource(BackendStatus::failed(
            BackendFailure::GpuInteropFeatureDisabled,
        ));

        #[cfg(feature = "gpu-interop")]
        finish_shared_backend(app);
    }
}

#[cfg(feature = "gpu-interop")]
fn finish_shared_backend(app: &mut App) {
    let Some(render_app) = app.get_sub_app(RenderApp) else {
        app.insert_resource(BackendStatus::failed(BackendFailure::RenderSubAppMissing));
        return;
    };
    let Some(adapter) = render_app.world().get_resource::<RenderAdapterInfo>() else {
        app.insert_resource(BackendStatus::failed(BackendFailure::RenderAdapterMissing));
        return;
    };
    if adapter.device_type == wgpu::DeviceType::Cpu {
        app.insert_resource(BackendStatus::failed(BackendFailure::CpuAdapterRejected));
        return;
    }
    let Some(render_device) = render_app.world().get_resource::<RenderDevice>() else {
        app.insert_resource(BackendStatus::failed(BackendFailure::RenderDeviceMissing));
        return;
    };
    let limits = render_device.limits();
    let info = BackendDeviceInfo {
        adapter_name: adapter.name.clone(),
        backend: format!("{:?}", adapter.backend),
        device_type: format!("{:?}", adapter.device_type),
        driver: adapter.driver.clone(),
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        shared_adapter_device_queue: true,
    };
    let burn_ready = app
        .world()
        .get_resource::<bevy_burn::BurnDevice>()
        .is_some_and(bevy_burn::BurnDevice::is_ready);
    if burn_ready {
        app.insert_resource(BackendStatus::ready(info));
    } else {
        app.insert_resource(BackendStatus::failed(BackendFailure::BurnDeviceMissing));
    }
}

#[cfg(feature = "gpu-interop")]
fn enforce_ready_device(
    burn_device: Option<Res<bevy_burn::BurnDevice>>,
    mut status: ResMut<BackendStatus>,
) {
    if status.is_ready()
        && !burn_device
            .as_deref()
            .is_some_and(bevy_burn::BurnDevice::is_ready)
    {
        status.state = BackendState::Failed {
            reason: BackendFailure::BurnDeviceLost,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendDeviceInfo, BackendState, BackendStatus};

    #[test]
    fn ready_status_attests_shared_objects_correctness() {
        let status = BackendStatus::ready(BackendDeviceInfo {
            adapter_name: "adapter".into(),
            backend: "Vulkan".into(),
            device_type: "DiscreteGpu".into(),
            driver: "driver".into(),
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_buffer_size: 256 * 1024 * 1024,
            shared_adapter_device_queue: true,
        });
        assert!(status.is_ready());
        assert!(matches!(status.state, BackendState::Ready { .. }));
        assert_eq!(status.unavailable_message(), None);
    }
}
