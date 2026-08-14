use cubecl_core::{Compiler, prelude::Visibility, server::KernelArguments};
use cubecl_core::{
    WgpuCompilationOptions,
    ir::{ElemType, UIntKind},
};
use cubecl_ir::{DeviceProperties, Type};
use wgpu::Features as WgpuFeatures;

use crate::WgslCompiler;

pub fn bindings(
    repr: &<WgslCompiler as Compiler>::Representation,
    args: &KernelArguments,
) -> (Vec<Visibility>, Option<Visibility>, bool) {
    let bindings = repr
        .buffers
        .iter()
        .map(|it| {
            if it.item.elem().is_atomic() {
                Visibility::ReadWrite
            } else {
                it.visibility
            }
        })
        .collect::<Vec<_>>();
    let meta = (!args.info.data.is_empty()).then_some(Visibility::Read);
    (bindings, meta, false)
}

pub async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: adapter
                .features()
                .difference(WgpuFeatures::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: limits,
            // The default is MemoryHints::Performance, which tries to do some bigger
            // block allocations. However, we already batch allocations, so we
            // can use MemoryHints::MemoryUsage to lower memory usage.
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            // SAFETY: Enabling experimental passthrough shaders.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        })
        .await
        .map_err(|err| {
            format!(
                "Unable to request the device with the adapter {:?}, err {:?}",
                adapter.get_info(),
                err
            )
        })
        .unwrap()
}

pub fn register_wgsl_features(
    device_features: WgpuFeatures,
    props: &mut cubecl_ir::DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props, device_features);
    if props.supports_type(ElemType::UInt(UIntKind::U64)) {
        comp_options.supports_u64 = true;
    }
}

pub fn register_types(props: &mut DeviceProperties, device_features: WgpuFeatures) {
    use cubecl_core::ir::{AddressType, ElemType, FloatKind, IntKind, StorageType};
    use cubecl_ir::features::*;

    props.register_address_type(AddressType::U32);

    let supported_types = [
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::Flex32),
        ElemType::Bool,
    ];

    let supported_atomic_types = [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all())
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ty)),
            AtomicUsage::LoadStore | AtomicUsage::Add,
        );
    }

    if device_features.contains(WgpuFeatures::SHADER_INT64) {
        props.register_type_usage(ElemType::Int(IntKind::I64), TypeUsage::all());
        props.register_type_usage(ElemType::UInt(UIntKind::U64), TypeUsage::all());
    }
    if device_features.contains(WgpuFeatures::SHADER_F64) {
        props.register_type_usage(ElemType::Float(FloatKind::F64), TypeUsage::all());
    }
    if device_features.contains(WgpuFeatures::SHADER_F16) {
        props.register_type_usage(ElemType::Float(FloatKind::F16), TypeUsage::all());
    }
    if device_features.contains(WgpuFeatures::SHADER_FLOAT32_ATOMIC) {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
            AtomicUsage::LoadStore | AtomicUsage::Add,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::profile::TimingMethod;
    use cubecl_core::ir::{FloatKind, IntKind, StorageType};
    use cubecl_ir::{HardwareProperties, MemoryDeviceProperties};

    fn properties() -> DeviceProperties {
        DeviceProperties::new(
            Default::default(),
            MemoryDeviceProperties {
                max_page_size: 1,
                alignment: 1,
            },
            HardwareProperties {
                load_width: 32,
                plane_size_min: 1,
                plane_size_max: 1,
                max_bindings: 1,
                max_shared_memory_size: 1,
                max_cube_count: (1, 1, 1),
                max_units_per_cube: 1,
                max_cube_dim: (1, 1, 1),
                num_streaming_multiprocessors: None,
                num_cpu_cores: None,
                num_tensor_cores: None,
                min_tensor_cores_dim: None,
                max_vector_size: 1,
            },
            TimingMethod::System,
        )
    }

    #[test]
    fn optional_wgsl_types_follow_enabled_device_features_correctness() {
        let mut downleveled = properties();
        register_types(&mut downleveled, WgpuFeatures::empty());
        assert!(!downleveled.supports_type(ElemType::Int(IntKind::I64)));
        assert!(!downleveled.supports_type(ElemType::Float(FloatKind::F16)));
        assert!(!downleveled.supports_type(ElemType::Float(FloatKind::F64)));
        assert!(
            downleveled
                .atomic_type_usage(Type::new(StorageType::Atomic(ElemType::Float(
                    FloatKind::F32
                ))))
                .is_empty()
        );

        let mut enabled = properties();
        register_types(
            &mut enabled,
            WgpuFeatures::SHADER_INT64
                | WgpuFeatures::SHADER_F16
                | WgpuFeatures::SHADER_F64
                | WgpuFeatures::SHADER_FLOAT32_ATOMIC,
        );
        assert!(enabled.supports_type(ElemType::Int(IntKind::I64)));
        assert!(enabled.supports_type(ElemType::Float(FloatKind::F16)));
        assert!(enabled.supports_type(ElemType::Float(FloatKind::F64)));
        assert!(
            !enabled
                .atomic_type_usage(Type::new(StorageType::Atomic(ElemType::Float(
                    FloatKind::F32
                ))))
                .is_empty()
        );
    }
}
