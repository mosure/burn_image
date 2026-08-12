//! Native verified-artifact Boogu inference on Burn WGPU.

#[path = "support/boogu_run.rs"]
mod boogu_run;

type Backend = burn_wgpu::Wgpu<f32, i32, u32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    boogu_run::run::<Backend, burn_boogu::BooguDenoiser<Backend>>(
        "burn-wgpu-native",
        || Ok(burn_boogu::require_native_wgpu_device()?),
        core::convert::identity,
        "portable-bounded-query-chunks",
    )
}
