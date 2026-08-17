import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES,
  ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
  ARTIFACT_TRANSPORT_LAYOUT_PATH,
  ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
  calculateArtifactManifestContentDigest,
  transportTelemetryFiles,
  validateArtifactBundleTransport,
} from "./artifact_transport_contract.mjs";

const digest = (marker) => marker.repeat(64);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function transportMetadata() {
  return {
    transport_layout_path: ARTIFACT_TRANSPORT_LAYOUT_PATH,
    transport_layout_schema: "1",
    transport_parts_required: "true",
    transport_part_target_bytes: String(ARTIFACT_TRANSPORT_TARGET_PART_BYTES),
    target_max_transport_shard_bytes: String(ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES),
    semantic_object_max_bytes: String(ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES),
    target_max_shard_bytes: String(ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES),
  };
}

async function partOnlyFixture() {
  const root = await mkdtemp(join(tmpdir(), "burn-image-transport-contract-"));
  await mkdir(join(root, "metadata"));
  await mkdir(join(root, "transport"));
  const configBytes = Buffer.from("{}\n");
  const partBytes = Buffer.from("bounded-part");
  const partSha256 = sha256(partBytes);
  const weightSha256 = digest("a");
  const partPath = `transport/${partSha256}.part`;
  const layout = {
    schema_version: 1,
    bundle: "fixture-bundle",
    profile: "fixture-profile",
    model: "fixture-model",
    model_revision: "fixture-revision",
    target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
    hard_max_part_bytes: ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
    objects: [
      {
        path: `objects/${weightSha256}.bpk`,
        size: partBytes.length,
        sha256: weightSha256,
        parts: [
          {
            path: partPath,
            offset: 0,
            size: partBytes.length,
            sha256: partSha256,
          },
        ],
      },
    ],
  };
  const sidecarBytes = Buffer.from(`${JSON.stringify(layout)}\n`);
  const manifest = {
    schema_version: 1,
    bundle: layout.bundle,
    profile: layout.profile,
    model: layout.model,
    model_revision: layout.model_revision,
    numeric_format: "f16",
    components: [{ id: "fixture-component", required: true }],
    dependencies: [],
    content_digest: null,
    metadata: transportMetadata(),
    files: [
      {
        path: `objects/${weightSha256}.bpk`,
        size: partBytes.length,
        sha256: weightSha256,
        role: "weights",
        component: "fixture-component",
        shard: null,
      },
      {
        path: "metadata/config.json",
        size: configBytes.length,
        sha256: sha256(configBytes),
        role: "config",
        component: null,
        shard: null,
      },
      {
        path: ARTIFACT_TRANSPORT_LAYOUT_PATH,
        size: sidecarBytes.length,
        sha256: sha256(sidecarBytes),
        role: "metadata",
        component: null,
        shard: null,
      },
    ],
  };
  manifest.content_digest = calculateArtifactManifestContentDigest(manifest);
  await Promise.all([
    writeFile(join(root, "metadata/config.json"), configBytes),
    writeFile(join(root, ARTIFACT_TRANSPORT_LAYOUT_PATH), sidecarBytes),
    writeFile(join(root, partPath), partBytes),
  ]);
  return { root, manifest, layout, partBytes, partPath };
}

test("matches the Rust manifest v1 digest vector correctness", () => {
  const manifest = {
    schema_version: 1,
    bundle: "example-bundle",
    profile: "f16-web",
    model: "owner/model",
    model_revision: "0123456789abcdef",
    numeric_format: "f16",
    components: [{ id: "transformer", required: true }],
    files: [
      {
        path: "transformer/model.bpk.part-000",
        size: 4,
        sha256: "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
        role: "weights",
        component: "transformer",
        shard: {
          index: 0,
          count: 2,
          chain_sha256: "de00eb6fbdca0b9f771dc0c428c84a6b3ad929bfdc0b7166b55226f1b328c5d3",
        },
      },
      {
        path: "transformer/model.bpk.part-001",
        size: 4,
        sha256: "27ecd0a598e76f8a2fd264d427df0a119903e8eae384e478902541756f089dd1",
        role: "weights",
        component: "transformer",
        shard: {
          index: 1,
          count: 2,
          chain_sha256: "8ad0b7b766fccc3564ab3cd9dd8dbc47ab9eba3d920f93e79c0ab35a854123e8",
        },
      },
    ],
    metadata: {},
    content_digest: "aaaaecdde41723f2203b30a40fb2ba6ee46b8c390b834e1d32c5746a8259abff",
  };

  assert.equal(
    calculateArtifactManifestContentDigest(manifest),
    manifest.content_digest,
  );
});

test("authenticates the sidecar and validates every part-only physical object", async () => {
  const fixture = await partOnlyFixture();
  try {
    const validation = await validateArtifactBundleTransport({
      bundleRoot: fixture.root,
      manifest: fixture.manifest,
    });
    assert.equal(validation.part_only, true);
    assert.equal(validation.sidecar.authenticated, true);
    assert.equal(validation.logical.file_count, 2);
    assert.equal(validation.logical.weight_file_count, 1);
    assert.equal(validation.manifest_declared.file_count, 3);
    assert.equal(validation.transport.unique_part_count, 1);
    assert.equal(validation.transport.unique_part_bytes, fixture.partBytes.length);
    assert.equal(validation.transport.reconstructed_bytes, fixture.partBytes.length);
    assert.equal(validation.transport.max_part_bytes, fixture.partBytes.length);
    assert.equal(validation.transport.target_part_bytes, 20 * 1024 * 1024);
    assert.equal(validation.transport.hard_max_part_bytes, 25_000_000);
    assert.equal(validation.transport.every_part_statted, true);
    assert.equal(
      validation.transport.part_sha256_policy,
      "verified-by-browser-runtime-before-use",
    );
    assert.deepEqual(transportTelemetryFiles(validation), [
      {
        path: fixture.partPath,
        size: fixture.partBytes.length,
        sha256: fixture.partPath.slice("transport/".length, -".part".length),
        component: "fixture-component",
        components: ["fixture-component"],
        logical_paths: [fixture.layout.objects[0].path],
        shared_physical_part: false,
        physical_transport_part: true,
      },
    ]);
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("attributes one deduplicated physical part to sorted logical components without double-counting bytes", async () => {
  const fixture = await partOnlyFixture();
  try {
    const sharedObject = {
      ...structuredClone(fixture.layout.objects[0]),
      path: `objects/${digest("e")}.bpk`,
      sha256: digest("e"),
    };
    fixture.layout.objects.push(sharedObject);
    fixture.manifest.files.splice(1, 0, {
      path: sharedObject.path,
      size: sharedObject.size,
      sha256: sharedObject.sha256,
      role: "weights",
      component: "another-component",
      shard: null,
    });
    fixture.manifest.components.push({ id: "another-component", required: true });
    const sidecarBytes = Buffer.from(`${JSON.stringify(fixture.layout)}\n`);
    const sidecar = fixture.manifest.files.find(
      (file) => file.path === ARTIFACT_TRANSPORT_LAYOUT_PATH,
    );
    sidecar.size = sidecarBytes.length;
    sidecar.sha256 = sha256(sidecarBytes);
    fixture.manifest.content_digest = calculateArtifactManifestContentDigest(fixture.manifest);
    await writeFile(join(fixture.root, ARTIFACT_TRANSPORT_LAYOUT_PATH), sidecarBytes);

    const validation = await validateArtifactBundleTransport({
      bundleRoot: fixture.root,
      manifest: fixture.manifest,
    });
    assert.equal(validation.transport.part_reference_count, 2);
    assert.equal(validation.transport.unique_part_count, 1);
    assert.equal(validation.transport.reconstructed_bytes, fixture.partBytes.length * 2);
    assert.equal(validation.transport.unique_part_bytes, fixture.partBytes.length);
    assert.deepEqual(transportTelemetryFiles(validation), [
      {
        path: fixture.partPath,
        size: fixture.partBytes.length,
        sha256: fixture.partPath.slice("transport/".length, -".part".length),
        component: "shared:another-component+fixture-component",
        components: ["another-component", "fixture-component"],
        logical_paths: [fixture.layout.objects[0].path, sharedObject.path].sort(),
        shared_physical_part: true,
        physical_transport_part: true,
      },
    ]);
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("rejects a present logical Burnpack in the canonical part-only tree", async () => {
  const fixture = await partOnlyFixture();
  try {
    await mkdir(join(fixture.root, "objects"));
    await writeFile(join(fixture.root, fixture.layout.objects[0].path), fixture.partBytes);
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: fixture.root, manifest: fixture.manifest }),
      /present in a part-only artifact tree/,
    );
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("rejects an unauthenticated sidecar and undeclared transport files", async () => {
  const fixture = await partOnlyFixture();
  try {
    const corrupted = Buffer.from(`${JSON.stringify({ ...fixture.layout, profile: "wrong" })}\n`);
    await writeFile(join(fixture.root, ARTIFACT_TRANSPORT_LAYOUT_PATH), corrupted);
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: fixture.root, manifest: fixture.manifest }),
      /must be a direct|sidecar SHA-256/,
    );
    await writeFile(
      join(fixture.root, ARTIFACT_TRANSPORT_LAYOUT_PATH),
      Buffer.from(`${JSON.stringify(fixture.layout)}\n`),
    );
    await writeFile(join(fixture.root, "transport/undeclared.part"), "extra");
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: fixture.root, manifest: fixture.manifest }),
      /missing or undeclared physical parts/,
    );
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("rejects a sidecar declaration changed without resealing the manifest", async () => {
  const fixture = await partOnlyFixture();
  try {
    const changedSidecar = Buffer.from(`${JSON.stringify(fixture.layout)} \n`);
    const sidecar = fixture.manifest.files.find(
      (file) => file.path === ARTIFACT_TRANSPORT_LAYOUT_PATH,
    );
    sidecar.size = changedSidecar.length;
    sidecar.sha256 = sha256(changedSidecar);
    await writeFile(join(fixture.root, ARTIFACT_TRANSPORT_LAYOUT_PATH), changedSidecar);
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: fixture.root, manifest: fixture.manifest }),
      /manifest content seal .* differs from declared/,
    );
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("rejects direct non-weight payloads above the exact physical CDN-object cap", async () => {
  const fixture = await partOnlyFixture();
  try {
    const config = fixture.manifest.files.find((file) => file.path === "metadata/config.json");
    config.size = ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES + 1;
    fixture.manifest.content_digest = calculateArtifactManifestContentDigest(fixture.manifest);
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: fixture.root, manifest: fixture.manifest }),
      /above the exact 25000000-byte physical CDN-object cap/,
    );
  } finally {
    await rm(fixture.root, { recursive: true, force: true });
  }
});

test("allows an old direct layout only through the explicit legacy fixture opt-in", async () => {
  const root = await mkdtemp(join(tmpdir(), "burn-image-legacy-transport-contract-"));
  const bytes = Buffer.from("legacy-direct-weight");
  const manifest = {
    schema_version: 1,
    bundle: "legacy-fixture",
    profile: "legacy",
    model: "legacy-model",
    model_revision: "legacy-revision",
    numeric_format: "f16",
    components: [{ id: "legacy-component", required: true }],
    dependencies: [],
    content_digest: null,
    metadata: {},
    files: [
      {
        path: "legacy.bpk",
        size: bytes.length,
        sha256: sha256(bytes),
        role: "weights",
        component: "legacy-component",
        shard: null,
      },
    ],
  };
  manifest.content_digest = calculateArtifactManifestContentDigest(manifest);
  try {
    await writeFile(join(root, "legacy.bpk"), bytes);
    await assert.rejects(
      validateArtifactBundleTransport({ bundleRoot: root, manifest }),
      /omits required metadata\/transport-layout.json/,
    );
    const validation = await validateArtifactBundleTransport({
      bundleRoot: root,
      manifest,
      allowLegacyDirectLayout: true,
    });
    assert.equal(validation.legacy_direct_layout, true);
    assert.equal(
      validation.policy,
      "explicit-legacy-direct-layout-no-browser-cache-shard-claim",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
