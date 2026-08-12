use burn::tensor::DType;
use burn_boogu::{BooguTask, BooguVariant, DmdSchedule, boogu_model_descriptor};

#[test]
fn public_release_contract_is_dtype_aware_correctness() {
    let descriptor = boogu_model_descriptor(BooguVariant::Image01Turbo);
    assert_eq!(descriptor.revision, burn_boogu::artifacts::TURBO_REVISION);
    assert_eq!(
        DmdSchedule::upstream_for_dtype(BooguTask::Generate, DType::F16).sigmas(),
        &[0.001_000_404_4, 0.250_732_42, 0.500_488_3, 0.75]
    );
}
