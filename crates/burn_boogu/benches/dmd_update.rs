use std::{hint::black_box, time::Instant};

use burn::tensor::{Tensor, TensorData};
use burn_boogu::{dmd_prediction, dmd_renoise};

type B = burn_ndarray::NdArray<f32>;

fn main() {
    const ITERATIONS: usize = 256;
    let device = Default::default();
    let values = (0..4 * 64 * 64)
        .map(|index| (index as f32 * 0.001).sin())
        .collect::<Vec<_>>();
    let latents =
        Tensor::<B, 4>::from_data(TensorData::new(values.clone(), [1, 4, 64, 64]), &device);
    let velocity = Tensor::<B, 4>::from_data(
        TensorData::new(
            values.iter().map(|value| value * 0.25).collect(),
            [1, 4, 64, 64],
        ),
        &device,
    );
    let noise = Tensor::<B, 4>::from_data(
        TensorData::new(values.iter().rev().copied().collect(), [1, 4, 64, 64]),
        &device,
    );

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let prediction = dmd_prediction(latents.clone(), velocity.clone(), 0.500_488_3);
        let output = dmd_renoise(prediction, noise.clone(), 0.75);
        black_box(output.into_data());
    }
    let elapsed = started.elapsed();
    println!(
        "dmd_update: {ITERATIONS} prediction+renoise updates in {:.6}s = {:.3} us/update",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
    );
}
