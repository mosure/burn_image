import { createHash } from "node:crypto";
import { lstat, readFile, readdir, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

export const ARTIFACT_TRANSPORT_LAYOUT_PATH = "metadata/transport-layout.json";
export const ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION = 1;
export const ARTIFACT_TRANSPORT_TARGET_PART_BYTES = 20 * 1024 * 1024;
export const ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES = 25_000_000;
export const ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES = 256 * 1024 * 1024;
export const MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES = 4 * 1024 * 1024;

const TRANSPORT_METADATA = Object.freeze({
  transport_layout_path: ARTIFACT_TRANSPORT_LAYOUT_PATH,
  transport_layout_schema: String(ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION),
  transport_parts_required: "true",
  transport_part_target_bytes: String(ARTIFACT_TRANSPORT_TARGET_PART_BYTES),
  target_max_transport_shard_bytes: String(ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES),
  semantic_object_max_bytes: String(ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES),
  target_max_shard_bytes: String(ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES),
});

const ARTIFACT_FILE_ROLE_DIGEST_TAG = Object.freeze({
  config: 0,
  tokenizer: 1,
  weights: 2,
  metadata: 3,
  other: 4,
});

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function exactSha256(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function exactSize(value, { positive = false } = {}) {
  return (
    Number.isSafeInteger(value) &&
    value >= (positive ? 1 : 0)
  );
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function updateU32(hasher, value, label) {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new Error(`${label} is not a canonical u32: ${JSON.stringify(value)}`);
  }
  const bytes = Buffer.allocUnsafe(4);
  bytes.writeUInt32LE(value);
  hasher.update(bytes);
}

function updateU64(hasher, value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} is not a canonical JavaScript-safe u64: ${JSON.stringify(value)}`);
  }
  const bytes = Buffer.allocUnsafe(8);
  bytes.writeBigUInt64LE(BigInt(value));
  hasher.update(bytes);
}

function updateString(hasher, value, label) {
  if (typeof value !== "string") {
    throw new Error(`${label} is not a string: ${JSON.stringify(value)}`);
  }
  const bytes = Buffer.from(value, "utf8");
  updateU64(hasher, bytes.length, `${label} UTF-8 length`);
  hasher.update(bytes);
}

function updateSha256(hasher, value, label) {
  if (!exactSha256(value)) {
    throw new Error(`${label} is not a lowercase SHA-256 digest: ${JSON.stringify(value)}`);
  }
  hasher.update(Buffer.from(value, "hex"));
}

function updateOptionalString(hasher, value, label) {
  if (value === null || value === undefined) {
    hasher.update(Buffer.from([0]));
    return;
  }
  hasher.update(Buffer.from([1]));
  updateString(hasher, value, label);
}

function numericFormatDigestName(value) {
  if (["f32", "f16", "bf16", "i8", "u8"].includes(value)) return value;
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 1 &&
    typeof value.other === "string"
  ) {
    return `other:${value.other}`;
  }
  throw new Error(`artifact numeric_format is invalid: ${JSON.stringify(value)}`);
}

/** Reproduce `ArtifactManifest::calculate_content_digest` without reading payload bytes. */
export function calculateArtifactManifestContentDigest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("artifact manifest is not an object");
  }
  if (![1, 2].includes(manifest.schema_version)) {
    throw new Error(`artifact manifest schema_version=${JSON.stringify(manifest.schema_version)}`);
  }
  if (!Array.isArray(manifest.components) || !Array.isArray(manifest.files)) {
    throw new Error("artifact manifest components/files are not arrays");
  }
  const dependencies = manifest.dependencies ?? [];
  if (!Array.isArray(dependencies)) {
    throw new Error("artifact manifest dependencies is not an array");
  }
  if (!manifest.metadata || typeof manifest.metadata !== "object" || Array.isArray(manifest.metadata)) {
    throw new Error("artifact manifest metadata is not an object");
  }

  const hasher = createHash("sha256");
  hasher.update(
    Buffer.from(
      manifest.schema_version === 1
        ? "burn_image.artifact_manifest.v1\0"
        : "burn_image.artifact_manifest.v2\0",
      "utf8",
    ),
  );
  updateU32(hasher, manifest.schema_version, "artifact schema_version");
  updateString(hasher, manifest.bundle, "artifact bundle");
  updateString(hasher, manifest.profile, "artifact profile");
  updateString(hasher, manifest.model, "artifact model");
  updateString(hasher, manifest.model_revision, "artifact model_revision");
  updateString(hasher, numericFormatDigestName(manifest.numeric_format), "artifact numeric_format");

  const components = [...manifest.components].sort((left, right) =>
    compareUtf8(String(left?.id ?? ""), String(right?.id ?? "")),
  );
  updateU64(hasher, components.length, "artifact component count");
  for (const component of components) {
    updateString(hasher, component?.id, "artifact component id");
    if (typeof component?.required !== "boolean") {
      throw new Error(`artifact component required flag is invalid: ${JSON.stringify(component)}`);
    }
    hasher.update(Buffer.from([component.required ? 1 : 0]));
  }

  if (manifest.schema_version >= 2) {
    const sortedDependencies = [...dependencies].sort((left, right) => {
      for (const field of ["role", "bundle", "profile", "model", "model_revision", "content_digest"]) {
        const order = compareUtf8(String(left?.[field] ?? ""), String(right?.[field] ?? ""));
        if (order !== 0) return order;
      }
      return 0;
    });
    updateU64(hasher, sortedDependencies.length, "artifact dependency count");
    for (const dependency of sortedDependencies) {
      for (const field of ["role", "bundle", "profile", "model", "model_revision"]) {
        updateString(hasher, dependency?.[field], `artifact dependency ${field}`);
      }
      updateSha256(hasher, dependency?.content_digest, "artifact dependency content_digest");
    }
  }

  const files = [...manifest.files].sort((left, right) =>
    compareUtf8(String(left?.path ?? ""), String(right?.path ?? "")),
  );
  updateU64(hasher, files.length, "artifact file count");
  for (const file of files) {
    updateString(hasher, file?.path, "artifact file path");
    updateU64(hasher, file?.size, `artifact file ${file?.path} size`);
    updateSha256(hasher, file?.sha256, `artifact file ${file?.path} SHA-256`);
    const roleTag = ARTIFACT_FILE_ROLE_DIGEST_TAG[file?.role];
    if (roleTag === undefined) {
      throw new Error(`artifact file ${file?.path} role is invalid: ${JSON.stringify(file?.role)}`);
    }
    hasher.update(Buffer.from([roleTag]));
    updateOptionalString(hasher, file?.component, `artifact file ${file?.path} component`);
    if (file?.shard === null || file?.shard === undefined) {
      hasher.update(Buffer.from([0]));
    } else {
      hasher.update(Buffer.from([1]));
      updateU32(hasher, file.shard.index, `artifact file ${file.path} shard index`);
      updateU32(hasher, file.shard.count, `artifact file ${file.path} shard count`);
      if (file.shard.chain_sha256 === null || file.shard.chain_sha256 === undefined) {
        hasher.update(Buffer.from([0]));
      } else {
        hasher.update(Buffer.from([1]));
        updateSha256(
          hasher,
          file.shard.chain_sha256,
          `artifact file ${file.path} shard chain SHA-256`,
        );
      }
    }
  }

  const metadata = Object.entries(manifest.metadata).sort(([left], [right]) =>
    compareUtf8(left, right),
  );
  updateU64(hasher, metadata.length, "artifact metadata count");
  for (const [key, value] of metadata) {
    updateString(hasher, key, "artifact metadata key");
    updateString(hasher, value, `artifact metadata ${key}`);
  }
  return hasher.digest("hex");
}

function validateManifestSeal(manifest) {
  const actual = calculateArtifactManifestContentDigest(manifest);
  if (actual !== manifest.content_digest) {
    throw new Error(
      `artifact manifest content seal ${actual} differs from declared ${manifest.content_digest}`,
    );
  }
}

function normalizedArtifactPath(value, label) {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    isAbsolute(value) ||
    value.includes("\\") ||
    value.split("/").some((component) => component === "" || component === "." || component === "..")
  ) {
    throw new Error(`${label} is not a canonical relative artifact path: ${JSON.stringify(value)}`);
  }
  return value;
}

function withinRoot(root, path) {
  const relativePath = relative(root, path);
  return relativePath === "" || (!relativePath.startsWith(`..${sep}`) && relativePath !== ".." && !isAbsolute(relativePath));
}

async function directFileMetadata(root, relativePath, expectedSize, label) {
  const path = resolve(root, normalizedArtifactPath(relativePath, label));
  if (!withinRoot(root, path)) throw new Error(`${label} escapes ${root}`);
  const metadata = await lstat(path).catch(() => null);
  if (!metadata?.isFile() || metadata.isSymbolicLink() || metadata.size !== expectedSize) {
    throw new Error(
      `${label} must be a direct ${expectedSize}-byte regular file; observed ${metadata?.size ?? "missing"}`,
    );
  }
  const canonical = await realpath(path);
  if (!withinRoot(root, canonical)) throw new Error(`${label} resolves outside ${root}`);
  return { path: canonical, size: metadata.size };
}

async function requireLogicalObjectAbsent(root, relativePath, label) {
  const path = resolve(root, normalizedArtifactPath(relativePath, label));
  if (!withinRoot(root, path)) throw new Error(`${label} escapes ${root}`);
  const metadata = await lstat(path).catch((error) => {
    if (error?.code === "ENOENT") return null;
    throw error;
  });
  if (metadata !== null) {
    throw new Error(`${label} is present in a part-only artifact tree`);
  }
}

function validateManifestFiles(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("artifact manifest is not an object");
  }
  if (!exactSha256(manifest.content_digest)) {
    throw new Error("artifact manifest is not sealed with a lowercase SHA-256 content digest");
  }
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error("artifact manifest has no files inventory");
  }
  const seen = new Set();
  for (const file of manifest.files) {
    const path = normalizedArtifactPath(file?.path, "manifest file path");
    if (
      seen.has(path) ||
      !exactSize(file?.size) ||
      !exactSha256(file?.sha256) ||
      !Object.hasOwn(ARTIFACT_FILE_ROLE_DIGEST_TAG, file?.role)
    ) {
      throw new Error(`artifact manifest has an invalid file entry: ${JSON.stringify(file)}`);
    }
    seen.add(path);
  }
}

function validateTransportDeclaration(manifest) {
  for (const [key, expected] of Object.entries(TRANSPORT_METADATA)) {
    if (manifest.metadata?.[key] !== expected) {
      throw new Error(
        `artifact manifest metadata.${key}=${JSON.stringify(manifest.metadata?.[key])}, expected ${JSON.stringify(expected)}`,
      );
    }
  }
  const sidecar = manifest.files.find((file) => file.path === ARTIFACT_TRANSPORT_LAYOUT_PATH);
  if (
    !sidecar ||
    sidecar.role !== "metadata" ||
    sidecar.component !== null ||
    sidecar.shard !== null ||
    sidecar.size > MAX_ARTIFACT_TRANSPORT_LAYOUT_BYTES
  ) {
    throw new Error(
      `${ARTIFACT_TRANSPORT_LAYOUT_PATH} is not a bounded direct manifest metadata file`,
    );
  }
  return sidecar;
}

function validateLayoutIdentity(manifest, layout) {
  const exactKeys = [
    "schema_version",
    "bundle",
    "profile",
    "model",
    "model_revision",
    "target_part_bytes",
    "hard_max_part_bytes",
    "objects",
  ];
  if (
    !layout ||
    typeof layout !== "object" ||
    Array.isArray(layout) ||
    JSON.stringify(Object.keys(layout).sort()) !== JSON.stringify([...exactKeys].sort())
  ) {
    throw new Error("artifact transport layout has missing or unknown top-level fields");
  }
  for (const field of ["bundle", "profile", "model", "model_revision"]) {
    if (layout[field] !== manifest[field]) {
      throw new Error(`artifact transport layout ${field} differs from its sealed manifest`);
    }
  }
  if (layout.schema_version !== ARTIFACT_TRANSPORT_LAYOUT_SCHEMA_VERSION) {
    throw new Error(`artifact transport layout schema_version=${layout.schema_version}`);
  }
  if (layout.target_part_bytes !== ARTIFACT_TRANSPORT_TARGET_PART_BYTES) {
    throw new Error(`artifact transport target is ${layout.target_part_bytes}, expected exact 20 MiB`);
  }
  if (layout.hard_max_part_bytes !== ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES) {
    throw new Error(
      `artifact transport hard maximum is ${layout.hard_max_part_bytes}, expected exact 25000000 bytes`,
    );
  }
  if (!Array.isArray(layout.objects) || layout.objects.length === 0) {
    throw new Error("artifact transport layout has no logical objects");
  }
}

function inventoryForFiles(files) {
  return files.reduce(
    (inventory, file) => {
      inventory.file_count += 1;
      inventory.bytes += file.size;
      inventory.max_file_bytes = Math.max(inventory.max_file_bytes, file.size);
      if (file.role === "weights") {
        inventory.weight_file_count += 1;
        inventory.weight_bytes += file.size;
        inventory.max_weight_file_bytes = Math.max(inventory.max_weight_file_bytes, file.size);
      }
      return inventory;
    },
    {
      file_count: 0,
      bytes: 0,
      max_file_bytes: 0,
      weight_file_count: 0,
      weight_bytes: 0,
      max_weight_file_bytes: 0,
    },
  );
}

function logicalInventory(manifest) {
  return inventoryForFiles(
    manifest.files.filter((file) => file.path !== ARTIFACT_TRANSPORT_LAYOUT_PATH),
  );
}

/**
 * Validate one local CDN bundle without reading every model byte.
 */
export async function validateArtifactBundleTransport({
  bundleRoot,
  manifest,
}) {
  validateManifestFiles(manifest);
  validateManifestSeal(manifest);
  const canonicalRoot = await realpath(bundleRoot);
  const rootMetadata = await stat(canonicalRoot);
  if (!rootMetadata.isDirectory()) throw new Error(`artifact bundle root is not a directory: ${canonicalRoot}`);

  const declaredSidecar = manifest.files.find(
    (file) => file.path === ARTIFACT_TRANSPORT_LAYOUT_PATH,
  );
  if (!declaredSidecar) {
    const hasTransportMetadata = Object.keys(TRANSPORT_METADATA).some((key) =>
      Object.hasOwn(manifest.metadata ?? {}, key),
    );
    throw new Error(
      hasTransportMetadata
        ? "artifact manifest has partial transport metadata without its sealed sidecar"
        : `artifact manifest omits required ${ARTIFACT_TRANSPORT_LAYOUT_PATH}`,
    );
  }

  const sidecar = validateTransportDeclaration(manifest);
  const sidecarFile = await directFileMetadata(
    canonicalRoot,
    sidecar.path,
    sidecar.size,
    "artifact transport sidecar",
  );
  const sidecarBytes = await readFile(sidecarFile.path);
  const sidecarSha256 = sha256(sidecarBytes);
  if (sidecarSha256 !== sidecar.sha256) {
    throw new Error(
      `artifact transport sidecar SHA-256 ${sidecarSha256} differs from sealed ${sidecar.sha256}`,
    );
  }
  let layout;
  try {
    layout = JSON.parse(sidecarBytes.toString("utf8"));
  } catch (error) {
    throw new Error(`artifact transport sidecar is not JSON: ${error}`);
  }
  validateLayoutIdentity(manifest, layout);

  const weightByPath = new Map(
    manifest.files
      .filter((file) => file.role === "weights")
      .map((file) => [file.path, file]),
  );
  const seenObjects = new Set();
  const uniqueParts = new Map();
  const partComponents = new Map();
  let previousObjectPath = null;
  let partReferenceCount = 0;
  let reconstructedBytes = 0;
  let maxPartBytes = 0;

  for (const object of layout.objects) {
    const objectPath = normalizedArtifactPath(object?.path, "transport logical object path");
    if (previousObjectPath !== null && previousObjectPath.localeCompare(objectPath) >= 0) {
      throw new Error(`artifact transport logical objects are not strictly path-sorted at ${objectPath}`);
    }
    previousObjectPath = objectPath;
    if (seenObjects.has(objectPath)) throw new Error(`duplicate transport logical object ${objectPath}`);
    seenObjects.add(objectPath);
    const logical = weightByPath.get(objectPath);
    if (
      !logical ||
      object.size !== logical.size ||
      object.sha256 !== logical.sha256 ||
      object.size > ARTIFACT_SEMANTIC_OBJECT_MAX_BYTES ||
      !Array.isArray(object.parts) ||
      object.parts.length === 0
    ) {
      throw new Error(`transport logical identity or parts differ for ${objectPath}`);
    }
    await requireLogicalObjectAbsent(canonicalRoot, objectPath, `logical weight ${objectPath}`);

    let expectedOffset = 0;
    for (const [index, part] of object.parts.entries()) {
      const partPath = normalizedArtifactPath(part?.path, `transport part for ${objectPath}`);
      if (
        part.offset !== expectedOffset ||
        !exactSize(part.size, { positive: true }) ||
        !exactSha256(part.sha256) ||
        partPath !== `transport/${part.sha256}.part` ||
        part.size > ARTIFACT_TRANSPORT_TARGET_PART_BYTES ||
        part.size > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES ||
        (index !== object.parts.length - 1 && part.size !== ARTIFACT_TRANSPORT_TARGET_PART_BYTES)
      ) {
        throw new Error(`invalid transport part ${partPath} for ${objectPath}`);
      }
      const prior = uniqueParts.get(partPath);
      if (prior && (prior.size !== part.size || prior.sha256 !== part.sha256)) {
        throw new Error(`conflicting transport part identity ${partPath}`);
      }
      uniqueParts.set(partPath, { size: part.size, sha256: part.sha256 });
      const components = partComponents.get(partPath) ?? new Set();
      if (typeof logical.component === "string") components.add(logical.component);
      partComponents.set(partPath, components);
      partReferenceCount += 1;
      reconstructedBytes += part.size;
      maxPartBytes = Math.max(maxPartBytes, part.size);
      expectedOffset += part.size;
    }
    if (expectedOffset !== logical.size) {
      throw new Error(
        `transport parts cover ${expectedOffset} bytes for ${objectPath}, expected ${logical.size}`,
      );
    }
  }
  if (seenObjects.size !== weightByPath.size) {
    const missing = [...weightByPath.keys()].filter((path) => !seenObjects.has(path));
    throw new Error(`artifact transport layout omits logical weights: ${missing.join(", ")}`);
  }

  const direct = { file_count: 0, bytes: 0, max_file_bytes: 0 };
  for (const file of manifest.files.filter((entry) => entry.role !== "weights")) {
    if (file.size > ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES) {
      throw new Error(
        `direct artifact ${file.path} has ${file.size} bytes, above the exact ${ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES}-byte physical CDN-object cap`,
      );
    }
    await directFileMetadata(canonicalRoot, file.path, file.size, `direct artifact ${file.path}`);
    direct.file_count += 1;
    direct.bytes += file.size;
    direct.max_file_bytes = Math.max(direct.max_file_bytes, file.size);
  }

  let uniquePartBytes = 0;
  for (const [partPath, identity] of uniqueParts) {
    await directFileMetadata(canonicalRoot, partPath, identity.size, `physical transport part ${partPath}`);
    uniquePartBytes += identity.size;
  }
  const transportDirectory = join(canonicalRoot, "transport");
  const observedPartNames = (await readdir(transportDirectory, { withFileTypes: true }))
    .map((entry) => {
      if (!entry.isFile() || entry.isSymbolicLink()) {
        throw new Error(`transport directory contains non-regular entry ${entry.name}`);
      }
      return entry.name;
    })
    .sort();
  const expectedPartNames = [...uniqueParts.keys()]
    .map((path) => path.slice("transport/".length))
    .sort();
  if (JSON.stringify(observedPartNames) !== JSON.stringify(expectedPartNames)) {
    throw new Error("transport directory contains missing or undeclared physical parts");
  }

  const telemetryEntries = [...partComponents.entries()].map(([path, components]) => {
    if (components.size === 0) {
      throw new Error(
        `transport part ${path} has no logical component attribution`,
      );
    }
    const sortedComponents = [...components].sort();
    const logicalPaths = layout.objects
      .filter((object) => object.parts.some((part) => part.path === path))
      .map((object) => object.path)
      .sort();
    return {
      path,
      size: uniqueParts.get(path).size,
      sha256: uniqueParts.get(path).sha256,
      component:
        sortedComponents.length === 1
          ? sortedComponents[0]
          : `shared:${sortedComponents.join("+")}`,
      components: sortedComponents,
      logical_paths: logicalPaths,
      shared_physical_part: sortedComponents.length > 1 || logicalPaths.length > 1,
      physical_transport_part: true,
    };
  }).sort((left, right) => left.path.localeCompare(right.path));

  return {
    policy: "manifest-sealed-part-only-browser-cache-transport-v1",
    part_only: true,
    sidecar: {
      path: sidecar.path,
      size: sidecar.size,
      sha256: sidecarSha256,
      authenticated: true,
    },
    logical: logicalInventory(manifest),
    manifest_declared: inventoryForFiles(manifest.files),
    direct,
    transport: {
      part_reference_count: partReferenceCount,
      unique_part_count: uniqueParts.size,
      reconstructed_bytes: reconstructedBytes,
      unique_part_bytes: uniquePartBytes,
      max_part_bytes: maxPartBytes,
      target_part_bytes: ARTIFACT_TRANSPORT_TARGET_PART_BYTES,
      hard_max_part_bytes: ARTIFACT_TRANSPORT_HARD_MAX_PART_BYTES,
      every_part_statted: true,
      part_sha256_policy: "verified-by-browser-runtime-before-use",
    },
    telemetry_entries: telemetryEntries,
  };
}

export function transportTelemetryFiles(validation) {
  if (validation?.part_only !== true || !Array.isArray(validation?.telemetry_entries)) {
    throw new Error("transport telemetry requires a validated part-only bundle");
  }
  return validation.telemetry_entries.map((entry) => ({ ...entry }));
}
