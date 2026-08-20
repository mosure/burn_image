import assert from "node:assert/strict";
import test from "node:test";

import {
  MODEL_RELEASES,
  canonicalModelRelease,
  configureModelReleaseSelector,
  modelReleaseSelectionState,
  modelReleaseUrl,
} from "../www/model_selector.mjs";

function selectorHarness(href) {
  let changeHandler = null;
  const note = { id: "model-release-note", textContent: "" };
  const select = {
    value: "",
    disabled: false,
    describedBy: null,
    setAttribute(name, value) {
      if (name === "aria-describedby") this.describedBy = value;
    },
    addEventListener(name, handler) {
      if (name === "change") changeHandler = handler;
    },
  };
  const assigned = [];
  const location = {
    href,
    assign(value) {
      assigned.push(value);
    },
  };
  return {
    select,
    note,
    location,
    assigned,
    change() {
      changeHandler();
    },
  };
}

test("canonical model release aliases remain stable", () => {
  assert.deepEqual(
    MODEL_RELEASES.map(({ value }) => value),
    ["turbo", "edit-turbo", "edit-turbo-1k5"],
  );
  assert.equal(canonicalModelRelease(null), "turbo");
  assert.equal(canonicalModelRelease("edit"), "edit-turbo");
  assert.equal(canonicalModelRelease("1k5"), "edit-turbo-1k5");
});

test("canonical release reload preserves safe query state and fragment", () => {
  const next = new URL(
    modelReleaseUrl(
      "https://example.test/app/?variant=turbo&profile=production&residency=low-vram&rendered-model-smoke=1#viewer",
      "edit-turbo-1k5",
    ),
  );
  assert.equal(next.searchParams.get("variant"), "edit-turbo-1k5");
  assert.equal(next.searchParams.get("profile"), "production");
  assert.equal(next.searchParams.get("residency"), "low-vram");
  assert.equal(next.searchParams.get("rendered-model-smoke"), "1");
  assert.equal(next.hash, "#viewer");
});

test("variant-specific residency is normalized to the bounded public selector", () => {
  for (const residency of [
    "low-vram-runtime-q8-denoiser",
    "low-vram-preloaded-ffn-gate-up-q8-denoiser",
    "qualification-f32",
    "unknown-policy",
  ]) {
    const next = new URL(
      modelReleaseUrl(
        `https://example.test/?variant=turbo&profile=production&residency=${residency}`,
        "edit-turbo",
      ),
    );
    assert.equal(next.searchParams.get("residency"), "low-vram");
  }
  const resident = new URL(
    modelReleaseUrl(
      "https://example.test/?variant=turbo&residency=resident",
      "edit-turbo",
    ),
  );
  assert.equal(resident.searchParams.get("residency"), "resident");
});

test("an omitted residency stays omitted for variant-aware defaults", () => {
  const next = new URL(
    modelReleaseUrl("https://example.test/?variant=turbo&profile=production", "edit-turbo"),
  );
  assert.equal(next.searchParams.has("residency"), false);
});

test("explicit custom artifacts lock the release selector", () => {
  const href =
    "https://example.test/?variant=turbo&artifacts=https%3A%2F%2Fcdn.test%2Fexact-turbo";
  const state = modelReleaseSelectionState(href);
  assert.equal(state.enabled, false);
  assert.match(state.reason, /pins one exact custom bundle/);
  assert.throws(() => modelReleaseUrl(href, "edit-turbo"), /pins one exact custom bundle/);
});

test("headless diagnostics cannot be mutated into another release", () => {
  const href = "https://example.test/?headless=parity&variant=edit-turbo-1k5";
  const state = modelReleaseSelectionState(href);
  assert.equal(state.enabled, false);
  assert.match(state.reason, /no-surface diagnostic/);
  assert.throws(() => modelReleaseUrl(href, "edit-turbo"), /no-surface diagnostic/);
});

test("unknown release targets fail rather than silently selecting Turbo", () => {
  assert.throws(
    () => modelReleaseUrl("https://example.test/?variant=turbo", "future-release"),
    /Unsupported model release/,
  );
});

test("accessible selector reloads a canonical page and describes the action", () => {
  const harness = selectorHarness(
    "https://example.test/?variant=turbo&profile=production&residency=low-vram",
  );
  const state = configureModelReleaseSelector(harness);
  assert.equal(state.enabled, true);
  assert.equal(harness.select.value, "turbo");
  assert.equal(harness.select.disabled, false);
  assert.equal(harness.select.describedBy, "model-release-note");
  assert.match(harness.note.textContent, /reloads this page/);

  harness.select.value = "edit-turbo";
  harness.change();
  assert.equal(harness.assigned.length, 1);
  assert.equal(new URL(harness.assigned[0]).searchParams.get("variant"), "edit-turbo");
});

test("accessible selector stays disabled for an exact custom bundle", () => {
  const harness = selectorHarness(
    "https://example.test/?variant=turbo&artifacts=https%3A%2F%2Fcdn.test%2Fexact-turbo",
  );
  const state = configureModelReleaseSelector(harness);
  assert.equal(state.enabled, false);
  assert.equal(harness.select.disabled, true);
  assert.match(harness.note.textContent, /pins one exact custom bundle/);
  assert.equal(harness.select.describedBy, "model-release-note");
});
