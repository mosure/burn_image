export const MODEL_RELEASES = Object.freeze([
  Object.freeze({ value: "turbo", label: "Turbo 1K generation" }),
  Object.freeze({ value: "edit-turbo", label: "Edit-Turbo 1K" }),
  Object.freeze({ value: "edit-turbo-1k5", label: "Edit-Turbo 1.5K" }),
]);

const RELEASE_VALUES = new Set(MODEL_RELEASES.map(({ value }) => value));
const GENERIC_RESIDENCIES = new Set(["low-vram", "resident"]);

export function canonicalModelRelease(value) {
  switch (value) {
    case null:
    case "":
    case "turbo":
      return "turbo";
    case "edit":
    case "edit-turbo":
      return "edit-turbo";
    case "1k5":
    case "edit-turbo-1k5":
      return "edit-turbo-1k5";
    default:
      return "turbo";
  }
}

export function modelReleaseSelectionState(href) {
  const url = new URL(href);
  const current = canonicalModelRelease(url.searchParams.get("variant"));
  if (url.searchParams.has("artifacts")) {
    return {
      current,
      enabled: false,
      reason:
        "Release switching is locked because artifacts= pins one exact custom bundle. Remove artifacts= to select a canonical release.",
    };
  }
  if (url.searchParams.has("headless")) {
    return {
      current,
      enabled: false,
      reason: "Release switching is unavailable while a no-surface diagnostic is active.",
    };
  }
  return {
    current,
    enabled: true,
    reason:
      "Changing release reloads this page and unloads the previous model. Warm GPU residency is the default; use ?residency=low-vram on smaller GPUs.",
  };
}

export function modelReleaseUrl(href, requestedRelease) {
  if (!RELEASE_VALUES.has(requestedRelease)) {
    throw new TypeError(`Unsupported model release: ${requestedRelease}`);
  }
  const state = modelReleaseSelectionState(href);
  if (!state.enabled) {
    throw new Error(state.reason);
  }

  const url = new URL(href);
  url.searchParams.set("variant", requestedRelease);
  const residency = url.searchParams.get("residency");
  if (residency !== null && !GENERIC_RESIDENCIES.has(residency)) {
    // Exact implementation selectors are variant-specific. Re-enter the public selector so the
    // target release resolves its own bounded low-VRAM policy.
    url.searchParams.set("residency", "low-vram");
  }
  return url.href;
}

export function configureModelReleaseSelector({ select, note, location }) {
  if (!select || !note || !location) {
    throw new TypeError("The model release selector requires select, note, and location objects");
  }
  const state = modelReleaseSelectionState(location.href);
  select.value = state.current;
  select.disabled = !state.enabled;
  note.textContent = state.reason;
  select.setAttribute("aria-describedby", note.id);
  select.addEventListener("change", () => {
    if (select.value === state.current) return;
    try {
      location.assign(modelReleaseUrl(location.href, select.value));
    } catch (error) {
      select.value = state.current;
      note.textContent = String(error);
    }
  });
  return state;
}
