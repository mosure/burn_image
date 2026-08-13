import assert from "node:assert/strict";
import test from "node:test";

import { attestCalibratedBrowserWebGpuScope } from "./wasm_browser_1k5_scope.mjs";

const expected = Object.freeze({
  chrome_product: "Chrome/151.0.7922.108",
  chrome_revision: "@4744b886309d987d292e43232776d2206cccb13d",
  adapter_vendor: "nvidia",
  adapter_architecture: "blackwell",
  adapter_device: "0x2bb1",
  adapter_description: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
  nvidia_driver_version: "610.43.02",
});

function matchingInputs() {
  return {
    browser: {
      product: expected.chrome_product,
      revision: expected.chrome_revision,
    },
    adapter: {
      vendor: expected.adapter_vendor,
      architecture: expected.adapter_architecture,
      device: expected.adapter_device,
      description: expected.adapter_description,
    },
    gpu: {
      provider: "nvidia-smi",
      gpu_indexes: [1],
      gpu_inventory: [
        {
          index: 0,
          uuid: "GPU-unrelated",
          name: "unrelated adapter",
          driver_version: "999.0",
        },
        {
          index: 1,
          uuid: "GPU-calibrated",
          name: expected.adapter_description,
          driver_version: expected.nvidia_driver_version,
        },
      ],
    },
  };
}

test("accepts the exact calibrated browser, adapter, and observed driver", () => {
  const input = matchingInputs();
  const result = attestCalibratedBrowserWebGpuScope(
    input.browser,
    input.adapter,
    input.gpu,
    expected,
  );
  assert.equal(result.validated, true);
  assert.deepEqual(result.validation_failures, []);
  assert.deepEqual(result.actual.observed_nvidia_gpus, [input.gpu.gpu_inventory[1]]);
});

for (const [label, target, field, mismatch] of [
  ["Chrome product", "browser", "product", "Chrome/152.0.0.0"],
  ["Chrome revision", "browser", "revision", "@different"],
  ["adapter vendor", "adapter", "vendor", "amd"],
  ["adapter architecture", "adapter", "architecture", "different"],
  ["adapter device", "adapter", "device", "0xffff"],
  ["adapter description", "adapter", "description", "different adapter"],
]) {
  test(`rejects a mismatched ${label}`, () => {
    const input = matchingInputs();
    input[target][field] = mismatch;
    const result = attestCalibratedBrowserWebGpuScope(
      input.browser,
      input.adapter,
      input.gpu,
      expected,
    );
    assert.equal(result.validated, false);
    assert.equal(result.validation_failures.length, 1);
  });
}

test("rejects a mismatched driver on the GPU index attributed to Chrome", () => {
  const input = matchingInputs();
  input.gpu.gpu_inventory[1].driver_version = "611.0";
  const result = attestCalibratedBrowserWebGpuScope(
    input.browser,
    input.adapter,
    input.gpu,
    expected,
  );
  assert.equal(result.validated, false);
  assert.match(result.validation_failures[0], /driver_version/);
});

test("rejects an inventory that cannot resolve the GPU index attributed to Chrome", () => {
  const input = matchingInputs();
  input.gpu.gpu_inventory.pop();
  const result = attestCalibratedBrowserWebGpuScope(
    input.browser,
    input.adapter,
    input.gpu,
    expected,
  );
  assert.equal(result.validated, false);
  assert.match(result.validation_failures[0], /expected exactly one/);
});
