const CI_WORKFLOW: &str = include_str!("workflows/ci.yml");
const PARITY_WORKFLOW: &str = include_str!("workflows/parity.yml");
const DEPLOY_WORKFLOW: &str = include_str!("workflows/deploy-pages.yml");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("source contract start marker is missing: {start}"))
        .split(end)
        .next()
        .unwrap_or_else(|| panic!("source contract end marker is missing: {end}"))
}

#[test]
fn release_evidence_uploads_only_allowlisted_files_correctness() {
    let uploads = section(
        PARITY_WORKFLOW,
        "      - name: Upload required release qualification evidence",
        "      - name: Require every release qualification gate",
    );

    let mut path_count = 0;
    for line in uploads.lines().map(str::trim) {
        if !line.starts_with("${{ runner.temp }}/") {
            continue;
        }
        path_count += 1;
        assert!(
            line.ends_with("/*.json")
                || line.ends_with("/*.log")
                || line.ends_with("/*.png")
                || line.ends_with("/downloads/*.png"),
            "evidence upload path is not an explicit JSON/log/PNG allowlist: {line}"
        );
    }

    assert!(
        path_count > 0,
        "evidence upload allowlist must not be empty"
    );
    assert!(!uploads.contains("burn-image-rendered-surface-profile-"));
    assert!(uploads.contains("if-no-files-found: error"));
    assert!(uploads.contains("always() && inputs.run_browser_f32_control_diagnostic"));
    assert!(uploads.contains("if-no-files-found: warn"));
}

#[test]
fn required_release_gate_aggregation_remains_fail_closed_correctness() {
    let gate = section(
        PARITY_WORKFLOW,
        "      - name: Require every release qualification gate",
        "          if ((failed != 0)); then",
    );
    for required in [
        "NATIVE_LOW_VRAM_OUTCOME",
        "LOW_VRAM_OUTCOME",
        "TURBO_FIRST_DMD_OUTCOME",
        "RENDERED_TURBO_OUTCOME",
        "RENDERED_TURBO_MULTI_REQUEST_OUTCOME",
        "NATIVE_TURBO_OUTPUT_OUTCOME",
        "NATIVE_BROWSER_OUTPUT_QUALITY_OUTCOME",
    ] {
        assert!(
            gate.contains(required),
            "required gate is missing: {required}"
        );
    }
    assert!(gate.contains("if [[ \"${!gate}\" != \"success\" ]]"));
}

#[test]
fn pages_authenticates_sealed_manifests_and_full_payloads_correctness() {
    assert!(DEPLOY_WORKFLOW.contains("timeout-minutes: 360"));
    assert!(DEPLOY_WORKFLOW.contains("--bin boogu-verify-artifacts"));
    assert!(DEPLOY_WORKFLOW.contains("--manifest-only \"$output\""));
    assert!(DEPLOY_WORKFLOW.contains(".sealed_manifest_verified == true"));
    for fetched_manifest in [
        "\"$MODEL_BASE_URL/manifest.json\"",
        "for role in qwen vae; do",
        "\"$EDIT_MODEL_BASE_URL/manifest.json\"",
        "\"$EDIT_1K5_MODEL_BASE_URL/manifest.json\"",
    ] {
        assert!(
            DEPLOY_WORKFLOW.contains(fetched_manifest),
            "canonical five-manifest fetch closure is missing: {fetched_manifest}"
        );
    }

    let authentication = section(
        DEPLOY_WORKFLOW,
        "          authenticate_remote_payloads() {",
        "          verify_manifest_contract \"$manifest_file\" \"$MODEL_BASE_URL\"",
    );
    for required in [
        "sort -u \"$payload_inventory\"",
        "remote payload URL has conflicting size/SHA-256 contracts",
        "((expected_size <= max_payload_bytes))",
        "curl --fail --silent --show-error",
        "--speed-time 120",
        "tee \"$payload_file\" | sha256sum",
        "\"$actual_size\" == \"$expected_size\"",
        "\"$actual_sha256\" == \"$expected_sha256\"",
    ] {
        assert!(
            authentication.contains(required),
            "remote authentication contract is missing: {required}"
        );
    }

    let last_manifest = DEPLOY_WORKFLOW
        .rfind("verify_manifest_contract \"$edit_1k5_manifest\"")
        .expect("1.5K manifest verification must remain present");
    let authenticate = DEPLOY_WORKFLOW
        .rfind("          authenticate_remote_payloads")
        .expect("full remote payload authentication must remain present");
    assert!(last_manifest < authenticate);
}

#[test]
fn package_outputs_do_not_pollute_the_native_target_cache_correctness() {
    let native = section(CI_WORKFLOW, "  native:\n", "\n  wasm:\n");
    assert!(native.contains("prefix-key: v1-rust-package-clean"));

    let package_step = native
        .split("      - name: Build and inspect publishable crate archives")
        .nth(1)
        .expect("package archive inspection step must remain present");
    for required in [
        "package_directory=\"$target_directory/package\"",
        "test \"$package_directory\" = \"$GITHUB_WORKSPACE/target/package\"",
        "mv -- \"$package_directory\" \"$inspected_package_directory\"",
        "test ! -e \"$package_directory\"",
        "test -d \"$inspected_package_directory\"",
    ] {
        assert!(
            package_step.contains(required),
            "package-cache cleanup contract is missing: {required}"
        );
    }

    let verified = package_step
        .find("verified package $archive")
        .expect("archive verification must remain present");
    let relocated = package_step
        .find("mv -- \"$package_directory\"")
        .expect("inspected package output relocation must remain present");
    assert!(verified < relocated);
}
