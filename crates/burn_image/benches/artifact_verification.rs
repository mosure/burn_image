use std::{hint::black_box, time::Instant};

use burn_image::{
    ArtifactFile, ArtifactFileRole, ArtifactPath, ArtifactVerifier, IntegrityPolicy, Sha256Digest,
};

fn main() {
    const BYTES: usize = 4 * 1024 * 1024;
    const ITERATIONS: usize = 64;
    const CHUNK: usize = 64 * 1024;

    let payload = (0..BYTES)
        .map(|index| (index.wrapping_mul(31) & 0xff) as u8)
        .collect::<Vec<_>>();
    let file = ArtifactFile {
        path: ArtifactPath::new("objects/benchmark.bpk").expect("benchmark path is valid"),
        size: payload.len() as u64,
        sha256: Sha256Digest::calculate(&payload),
        role: ArtifactFileRole::Weights,
        component: None,
        shard: None,
    };

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let mut verifier = ArtifactVerifier::new(&file, IntegrityPolicy::RequireSha256);
        for chunk in payload.chunks(CHUNK) {
            verifier.update(black_box(chunk)).expect("bounded update");
        }
        black_box(verifier.finish().expect("exact digest"));
    }
    let elapsed = started.elapsed();
    let total_bytes = (BYTES * ITERATIONS) as f64;
    let mib_per_second = total_bytes / (1024.0 * 1024.0) / elapsed.as_secs_f64();
    println!(
        "artifact_verification: {ITERATIONS} x {BYTES} bytes in {:.6}s = {:.2} MiB/s",
        elapsed.as_secs_f64(),
        mib_per_second
    );
}
