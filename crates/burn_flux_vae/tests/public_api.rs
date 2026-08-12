use burn_flux_vae::AutoencoderKlConfig;

#[test]
fn reusable_flux_config_is_boogu_independent_correctness() {
    let config = AutoencoderKlConfig::flux1();
    config.validate().unwrap();
    assert_eq!(config.latent_channels, 16);
}
