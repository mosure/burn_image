//! Native verified-artifact Boogu inference on Burn CUDA.

#[path = "support/boogu_run.rs"]
mod boogu_run;

type Backend = burn::backend::Cuda<f32, i32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    boogu_run::run::<Backend, burn_boogu::NativeCudaPaddedBlackboxDenoiser>(
        "burn-cuda-native-padded-blackbox-experimental",
        || Ok(burn::backend::cuda::CudaDevice::default()),
        |denoiser| burn_boogu::NativeCudaPaddedBlackboxDenoiser::new(denoiser).with_num_planes(4),
        "experimental-cuda-forced-padded-blackbox-16x16x16",
    )
}
