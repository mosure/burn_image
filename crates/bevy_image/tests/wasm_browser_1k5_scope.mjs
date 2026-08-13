function recordExactMatch(failures, path, actual, expected) {
  if (actual !== expected) {
    failures.push(
      `${path}=${JSON.stringify(actual)}, expected calibrated ${JSON.stringify(expected)}`,
    );
  }
}

/**
 * Bind a browser parity result to the exact runtime stack used to calibrate its numerical
 * envelope. NVIDIA inventory entries are checked only for GPU indexes that pmon attributed to
 * Chrome, so unrelated installed adapters cannot invalidate otherwise matching evidence.
 */
export function attestCalibratedBrowserWebGpuScope(
  browser,
  adapter,
  nativeGpuAttestation,
  expected,
) {
  const failures = [];
  const actual = {
    chrome_product: browser?.product ?? null,
    chrome_revision: browser?.revision ?? null,
    adapter_vendor: adapter?.vendor ?? null,
    adapter_architecture: adapter?.architecture ?? null,
    adapter_device: adapter?.device ?? null,
    adapter_description: adapter?.description ?? null,
    nvidia_smi_provider: nativeGpuAttestation?.provider ?? null,
    observed_gpu_indexes: Array.isArray(nativeGpuAttestation?.gpu_indexes)
      ? nativeGpuAttestation.gpu_indexes
      : null,
    observed_nvidia_gpus: [],
  };

  for (const [field, path] of [
    ["chrome_product", "browser.product"],
    ["chrome_revision", "browser.revision"],
    ["adapter_vendor", "native_webgpu_adapter.vendor"],
    ["adapter_architecture", "native_webgpu_adapter.architecture"],
    ["adapter_device", "native_webgpu_adapter.device"],
    ["adapter_description", "native_webgpu_adapter.description"],
  ]) {
    recordExactMatch(failures, path, actual[field], expected[field]);
  }
  recordExactMatch(
    failures,
    "native_gpu_attestation.provider",
    actual.nvidia_smi_provider,
    "nvidia-smi",
  );

  const inventory = nativeGpuAttestation?.gpu_inventory;
  const indexes = actual.observed_gpu_indexes;
  if (!Array.isArray(inventory)) {
    failures.push("native_gpu_attestation.gpu_inventory is not an array");
  }
  if (
    !Array.isArray(indexes) ||
    indexes.length === 0 ||
    indexes.some((index) => !Number.isSafeInteger(index) || index < 0) ||
    new Set(indexes).size !== indexes.length
  ) {
    failures.push(
      "native_gpu_attestation.gpu_indexes must contain unique non-negative GPU indexes attributed to Chrome",
    );
  } else if (Array.isArray(inventory)) {
    for (const index of indexes) {
      const matches = inventory.filter((gpu) => gpu?.index === index);
      if (matches.length !== 1) {
        failures.push(
          `native_gpu_attestation.gpu_inventory has ${matches.length} entries for Chrome GPU index ${index}; expected exactly one`,
        );
        continue;
      }
      const gpu = matches[0];
      actual.observed_nvidia_gpus.push({
        index: gpu.index,
        uuid: gpu.uuid ?? null,
        name: gpu.name ?? null,
        driver_version: gpu.driver_version ?? null,
      });
      recordExactMatch(
        failures,
        `native_gpu_attestation.gpu_inventory[index=${index}].driver_version`,
        gpu.driver_version,
        expected.nvidia_driver_version,
      );
    }
  }

  return {
    policy: "exact-calibrated-browser-webgpu-stack",
    portability: "no-cross-browser-adapter-or-driver-portability-claim",
    expected: { ...expected },
    actual,
    validation_failures: failures,
    validated: failures.length === 0,
  };
}
