# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device geniex-bench scorecard run for QDC Android phones.

Mirrors run_android.py's robustness pattern: network preflight, single
retry per cell, and explicit on-device output check.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest

# QDC stages the extracted test package under /qdc/appium; AWS Device
# Farm's custom test environment extracts it under
# $DEVICEFARM_TEST_PACKAGE_PATH instead (see aws_test_spec.yaml, which sets
# the override).
HOST_ARTIFACT_ROOT = os.environ.get("QAIHM_HOST_ARTIFACT_ROOT", "/qdc/appium")
HOST_QAIRT_BUNDLES = f"{HOST_ARTIFACT_ROOT}/qairt_bundles"
HOST_ROWS = f"{HOST_ARTIFACT_ROOT}/matrix_rows.txt"
HOST_CHIPSET = f"{HOST_ARTIFACT_ROOT}/chipset.txt"
HOST_PROMPTS = f"{HOST_ARTIFACT_ROOT}/prompts"
HOST_STAGE = f"{HOST_ARTIFACT_ROOT}/_stage"
HOST_BUNDLE = f"{HOST_STAGE}/bundle"

DEVICE_BUNDLE = "/data/local/tmp/pkg-geniex"
DEVICE_QAIRT_BUNDLES = f"{DEVICE_BUNDLE}/qairt_bundles"
DEVICE_MM_CACHE = "/data/local/tmp/geniex-cache"
# Must not share its name with the host-side log-dir placeholder: that replace
# would also rewrite every f-string reference to it into a relative path.
DEVICE_LOGS = "/data/local/tmp/{DEVICE_LOGS_DIR}"
DEVICE_RESULTS = f"{DEVICE_LOGS}/results"
DEVICE_PROMPTS = "/data/local/tmp/eval_prompts"
DEVICE_EVAL_OUT = f"{DEVICE_LOGS}/geniex_eval_outputs.txt"
DEVICE_EVAL_ERR = f"{DEVICE_LOGS}/geniex_eval_stderr.txt"

CTXS = tuple(int(c) for c in "{CTX_LIST}".split(","))
ANDROID_BENCH_URL = "{ANDROID_BENCH_URL}"
# Presigned S3 GET URL for the qairt bundle (empty when the plugin isn't
# qairt, or when the caller has no S3 asset -- e.g. local/dev runs), so the
# device curls its weights directly instead of the host pushing them via adb,
# bypassing AWS Device Farm's ~4GB artifact-upload cap.
QAIRT_BUNDLE_URL = "{QAIRT_BUNDLE_URL}"
PLUGIN = "{PLUGIN}"
N_GEN = int("{N_GEN}")
EVAL_CTX = int("{EVAL_CTX}")
EVAL_N_GEN = int("{EVAL_N_GEN}")
EVAL_TIMEOUT_S = int("{EVAL_TIMEOUT_S}")
EVAL_SLEEP_S = int("{EVAL_SLEEP_S}")
# Placeholder substituted with the literal "1"/"0" host-side; the string
# comparison keeps the template statically typed (a bare `= {RUN_PERF}` reads as
# a self-referential set literal to mypy).
RUN_PERF = "{RUN_PERF}" == "1"  # noqa: PLR0133
RUN_EVAL = "{RUN_EVAL}" == "1"  # noqa: PLR0133
# QAIRT SDK to pass via --qairt-lib (e.g. "2.45.0.260326"); empty keeps the
# bundled QAIRT.
QAIRT_VERSION = "{QAIRT_VERSION}"
DEVICE_QAIRT_ZIP = "/data/local/tmp/qairt.zip"
# The SDK zip's top-level entry is qairt/<version>/, so extract into its parent.
DEVICE_QAIRT_EXTRACT = "/data/local/tmp"
DEVICE_QAIRT_ROOT = f"{DEVICE_QAIRT_EXTRACT}/qairt"


def adb(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    # adb shell drops the remote exit code; recover it via __RC__ trailer.
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    stdout, rc = raw.stdout, raw.returncode
    lines = stdout.rstrip("\n").split("\n") if stdout else []
    if lines and lines[-1].startswith("__RC__:"):
        try:
            rc = int(lines[-1][7:])
            stdout = "\n".join(lines[:-1]) + "\n"
        except ValueError:
            pass
    print(stdout)
    result = subprocess.CompletedProcess(raw.args, rc, stdout=stdout)
    if check:
        assert rc == 0, f"adb command failed (exit {rc}): {cmd}"
    return result


def _preflight_url(url: str) -> None:
    # QDC phones occasionally boot with degraded wifi; fail fast instead
    # of letting stage_bundle()/_fetch_qairt_bundle_ondevice() hang silently
    # on a stalled fetch. A presigned S3 GET URL's signature is bound to the
    # HTTP method, so a HEAD (or a bucket-root probe) gets rejected with 403
    # regardless of connectivity -- use `-r 0-0` to keep this a GET while
    # only pulling the first byte.
    preflight = subprocess.run(
        [
            "adb",
            "shell",
            f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time 15 -r 0-0 '{url}'",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    http_code = preflight.stdout.strip()
    if preflight.returncode == 0 and http_code.startswith(("2", "3")):
        return
    if preflight.returncode == 0 and http_code.startswith("4"):
        # The device has network; surface S3's error body (e.g. ExpiredToken).
        body = subprocess.run(
            ["adb", "shell", f"curl -sS --max-time 15 '{url}' | head -c 600"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        pytest.fail(
            f"Server rejected {url} (http_code={http_code!r}); device network is "
            f"fine, the URL is likely expired or invalid. Response: {body!r}"
        )
    pytest.fail(
        f"Device cannot reach {url} (rc={preflight.returncode}, "
        f"http_code={http_code!r}, stderr={preflight.stderr!r}). "
        "Likely QDC device-side wifi failure — file a QDC infra ticket and re-run."
    )


def _preflight_network() -> None:
    _preflight_url(ANDROID_BENCH_URL)
    if QAIRT_BUNDLE_URL:
        _preflight_url(QAIRT_BUNDLE_URL)


def stage_bundle() -> None:
    if os.path.exists(os.path.join(HOST_BUNDLE, "bin", "geniex-bench")):
        return
    os.makedirs(HOST_STAGE, exist_ok=True)

    print(f"Fetching {ANDROID_BENCH_URL}")
    with urllib.request.urlopen(ANDROID_BENCH_URL) as resp:
        bench_tgz = resp.read()
    with tarfile.open(fileobj=io.BytesIO(bench_tgz), mode="r:gz") as tf:
        members = tf.getmembers()
        top = members[0].name.split("/", 1)[0] if members else ""
        for m in members:
            if not m.name.startswith(top + "/") or m.name == top + "/":
                continue
            rel = m.name[len(top) + 1 :]
            dst = os.path.join(HOST_BUNDLE, rel)
            real_dst = os.path.realpath(dst)
            real_base = os.path.realpath(HOST_BUNDLE)
            if not real_dst.startswith(real_base + os.sep):
                raise ValueError(f"Refusing unsafe tar member path: {m.name!r}")
            if m.isdir():
                os.makedirs(dst, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            os.chmod(dst, m.mode | 0o400)


def push_bundle() -> None:
    # QDC reflashes the phone every session, so always push.
    stage_bundle()
    subprocess.run(["adb", "push", HOST_BUNDLE, DEVICE_BUNDLE], check=True)
    adb(f"find {DEVICE_BUNDLE}/bin -type f -exec chmod 755 {{}} +")
    adb(f"cp {DEVICE_BUNDLE}/lib/qairt/htp-files/*.so {DEVICE_BUNDLE}/lib/")
    adb(f"cp {DEVICE_BUNDLE}/lib/llama_cpp/*.so {DEVICE_BUNDLE}/lib/")


def _fetch_qairt_bundle_ondevice(bundle_name: str) -> str:
    """Curl+unzip the presigned qairt bundle directly on-device.

    Mirrors run_android.py's genie_bundle_download_block: bypasses AWS
    Device Farm's ~4GB artifact-upload cap by having the device pull weights
    straight from S3 instead of the host adb-pushing them. The zip's own
    top-level folder is named after the long release-asset name (see
    ASSET_CONFIG.get_release_asset_name), not the bare model_id that
    _rewrite_matrix_for_qairt_bundles (geniex/jobs.py) already baked into
    matrix_rows.txt -- so the extracted dir is moved to `bundle_name`
    (the bare model_id) to match what the matrix/eval model refs expect.
    Returns `bundle_name` unchanged, for the caller's convenience.
    """
    adb(f"rm -rf {DEVICE_QAIRT_BUNDLES} /data/local/tmp/_qairt_bundle_dl")
    adb(f"mkdir -p {DEVICE_QAIRT_BUNDLES} /data/local/tmp/_qairt_bundle_dl")
    adb(
        "curl -L --fail --max-time 1800 --retry 3 --retry-delay 5 "
        f"--output /data/local/tmp/qairt_bundle.zip '{QAIRT_BUNDLE_URL}'"
    )
    # Android's toybox unzip can return a nonzero warning exit even on a fully
    # successful extract (see run_android.py's qairt.zip handling); don't
    # trust the exit code, verify the extracted dir exists instead, retrying
    # the extract once if it doesn't.
    adb(
        "unzip -q /data/local/tmp/qairt_bundle.zip -d /data/local/tmp/_qairt_bundle_dl",
        check=False,
    )
    find = adb(
        "find /data/local/tmp/_qairt_bundle_dl -mindepth 1 -maxdepth 1 -type d "
        "| head -n1"
    )
    extracted = find.stdout.strip()
    if not extracted:
        adb("rm -rf /data/local/tmp/_qairt_bundle_dl")
        adb("mkdir -p /data/local/tmp/_qairt_bundle_dl")
        adb(
            "unzip -q /data/local/tmp/qairt_bundle.zip "
            "-d /data/local/tmp/_qairt_bundle_dl",
            check=False,
        )
        find = adb(
            "find /data/local/tmp/_qairt_bundle_dl -mindepth 1 -maxdepth 1 -type d "
            "| head -n1"
        )
        extracted = find.stdout.strip()
    adb("rm -f /data/local/tmp/qairt_bundle.zip")
    assert extracted, "qairt bundle zip had no top-level directory"
    adb(f"mv '{extracted}' {DEVICE_QAIRT_BUNDLES}/{bundle_name}")
    adb("rm -rf /data/local/tmp/_qairt_bundle_dl")
    return bundle_name


def fetch_qairt_sdk() -> str:
    """Download QAIRT_VERSION's SDK on-device; return the --qairt-lib flag for it.

    Returns an empty string (no flag) when the bundled QAIRT should be used.
    """
    if PLUGIN != "qairt" or not QAIRT_VERSION:
        return ""
    print(f"=== Using QAIRT {QAIRT_VERSION} ===")
    sdk = f"{DEVICE_QAIRT_ROOT}/{QAIRT_VERSION}"
    url = (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip"
    )
    adb(
        "curl -L -J --fail --max-time 300 --retry 3 --retry-delay 5 "
        f"--output {DEVICE_QAIRT_ZIP} {url}"
    )
    adb(f"rm -rf {DEVICE_QAIRT_ROOT}")
    # toybox unzip can exit nonzero on a successful extract, so check=False and
    # verify the extracted dir instead of trusting its exit code.
    adb(f"unzip -q {DEVICE_QAIRT_ZIP} -d {DEVICE_QAIRT_EXTRACT}", check=False)
    adb(f"rm -f {DEVICE_QAIRT_ZIP}", check=False)
    adb(f"test -d {sdk}/lib")
    return f"--qairt-lib {sdk}"


def _run_bench(
    ctx: int, env: str, tsv_path: str, chipset: str, qairt_lib_flag: str
) -> int:
    size_flags = f"-c {ctx + N_GEN} -p {ctx - N_GEN} -n {N_GEN}"
    cmd = (
        f"cd {DEVICE_BUNDLE} && {env} ./bin/geniex-bench "
        f"--matrix-file {tsv_path} --output-json-dir {DEVICE_RESULTS} -r 3 "
        f"{qairt_lib_flag} {size_flags} "
        f"--mm-data-dir {DEVICE_MM_CACHE} --chipset '{chipset}' "
        f"2>>{DEVICE_LOGS}/geniex_bench_stderr.log"
    )
    res = adb(cmd, check=False)
    if res.returncode == 0:
        return 0
    print(f"geniex-bench ctx={ctx} failed (rc={res.returncode}); retrying once.")
    return adb(cmd, check=False).returncode


def _cleanup_device() -> None:
    # Drop per-job state (dedicated-pool devices are reused across jobs; leftover
    # bundles / caches / matrix TSVs would leak into the next tenant's run).
    # device_logs is left alone so retrieval still sees results/logs.
    adb(
        f"rm -rf {DEVICE_BUNDLE} {DEVICE_MM_CACHE} {DEVICE_QAIRT_ROOT} "
        f"{DEVICE_QAIRT_ZIP}",
        check=False,
    )
    adb("rm -f /data/local/tmp/matrix-*.tsv", check=False)


def _count_eval_markers() -> int:
    """Count the ===EVAL_IDX=== markers currently in the on-device eval output.

    Returns
    -------
    int
        The marker count, or 0 if the file is absent or adb is unreachable.
    """
    n = adb(f"grep -c '===EVAL_IDX_' {DEVICE_EVAL_OUT}", check=False)
    try:
        return int(n.stdout.strip())
    except ValueError:
        # grep found nothing, or adb itself is gone and returned no count.
        return 0


def _run_eval(
    env: str, model_ref: str, device_alias: str, chipset: str, qairt_lib_flag: str
) -> tuple[int, int, int]:
    """Run `geniex-bench --accuracy` once per staged prompt, collecting stdout.

    Output for each prompt is marked with ===EVAL_IDX_NNN=== so the host can
    split and grade it.

    Returns ``(ran, collected, expected)``. The first two diverge when the
    device drops off adb mid-run:

    - ``ran`` counts prompts whose geniex-bench invocation exited 0. A
      context-length overflow keeps its partial [gen] output and counts as run.
    - ``collected`` counts the ===EVAL_IDX=== markers actually readable back
      from the device, i.e. what the host will be able to retrieve.
    - ``expected`` is the staged prompt count, returned so the caller compares
      against the set this loop iterated rather than re-globbing.
    """
    assert os.path.isdir(HOST_PROMPTS), f"eval requested but {HOST_PROMPTS} missing"
    assert os.path.isfile(os.path.join(HOST_PROMPTS, "system_prompt.txt")), (
        f"eval requested but {HOST_PROMPTS}/system_prompt.txt missing"
    )
    prompt_files = sorted(
        f for f in os.listdir(HOST_PROMPTS) if f.startswith("prompt_")
    )
    adb(f"mkdir -p {DEVICE_PROMPTS}")
    subprocess.run(["adb", "push", f"{HOST_PROMPTS}/.", DEVICE_PROMPTS], check=True)
    # check=False so an unreachable device is reported by the eval loop below
    # rather than by an assert on cleanup. Verify the post-state instead: on a
    # reused cell (_cleanup_device spares device_logs) a surviving marker would
    # inflate the completeness count and mask an incomplete harvest.
    adb(f"rm -f {DEVICE_EVAL_OUT} {DEVICE_EVAL_ERR}", check=False)
    stale = _count_eval_markers()
    assert stale == 0, (
        f"{DEVICE_EVAL_OUT} still holds {stale} ===EVAL_IDX=== marker(s) after "
        "cleanup; a previous job's output would be mixed into this run's results."
    )
    eval_ran = 0
    for fn in prompt_files:
        idx = fn[len("prompt_") : -len(".txt")]
        pf = f"{DEVICE_PROMPTS}/{fn}"
        perr = f"{DEVICE_LOGS}/geniex_eval_prompt_stderr.txt"
        pout = f"{DEVICE_LOGS}/geniex_eval_prompt_stdout.txt"
        # Each attempt writes to fresh per-prompt temp files on device; the
        # accepted attempt's stdout is concatenated under the ===EVAL_IDX===
        # marker exactly once so retries do not concatenate two generations at
        # the grader. timeout kills a crashed/hung DSP run so the loop advances;
        # a context-length overflow is a legit long generation (partial output
        # kept), so force rc=0 and skip retry.
        cmd = (
            f"cd {DEVICE_BUNDLE} && "
            f"rm -f {pout} {perr} && "
            f"{env} timeout {EVAL_TIMEOUT_S} ./bin/geniex-bench --plugin {PLUGIN} "
            f"--device {device_alias} -m {model_ref} {qairt_lib_flag} "
            f"--accuracy --prompt-file {pf} "
            f'--system-prompt "$(cat {DEVICE_PROMPTS}/system_prompt.txt)" --no-think '
            f"-c {EVAL_CTX} -n {EVAL_N_GEN} --mm-data-dir {DEVICE_MM_CACHE} "
            f"--chipset '{chipset}' >{pout} 2>{perr}; "
            f"rc=$?; grep -q 'Context length exceeded' {perr} && rc=0; exit $rc"
        )
        # Retry a random device flake once; the second run's fresh pout/perr
        # overwrite the first attempt's output.
        res = adb(cmd, check=False)
        if res.returncode != 0:
            print(f"eval idx {idx} failed (rc={res.returncode}); retrying once")
            res = adb(cmd, check=False)
        if res.returncode == 0:
            eval_ran += 1
        else:
            print(f"eval idx {idx} failed (rc={res.returncode})")
        # Emit marker + accepted attempt's output once, then wipe the tmpfiles.
        adb(f"echo '===EVAL_IDX_{idx}===' >> {DEVICE_EVAL_OUT}", check=False)
        adb(f"echo '===EVAL_IDX_{idx}===' >> {DEVICE_EVAL_ERR}", check=False)
        adb(f"cat {pout} >> {DEVICE_EVAL_OUT} 2>/dev/null", check=False)
        adb(f"cat {perr} >> {DEVICE_EVAL_ERR} 2>/dev/null", check=False)
        adb(f"rm -f {pout} {perr}", check=False)
        adb(f"sleep {EVAL_SLEEP_S}", check=False)
    collected = _count_eval_markers()
    print(
        f"eval done ({collected} prompts collected, {eval_ran} ran, "
        f"{len(prompt_files)} expected)"
    )
    return eval_ran, collected, len(prompt_files)


def test_scorecard() -> None:
    _preflight_network()
    push_bundle()
    qairt_lib_flag = fetch_qairt_sdk()
    # QDC reuses device cells across jobs; wipe stale cell JSONs from a prior
    # run so compute_metrics doesn't ingest another model/plugin's results.
    adb(f"rm -rf {DEVICE_RESULTS}", check=False)
    adb(f"mkdir -p {DEVICE_MM_CACHE} {DEVICE_RESULTS}")
    try:
        chipset = Path(HOST_CHIPSET).read_text().strip()
        rows = [r for r in Path(HOST_ROWS).read_text().splitlines() if r.strip()]
        assert rows, "no model rows produced"

        bundle_name: str | None = None
        if PLUGIN == "qairt" and QAIRT_BUNDLE_URL:
            # Bundle must have consistent name with name in matrix.
            model_id_for_bundle = rows[0].split("|", 1)[0]
            bundle_name = _fetch_qairt_bundle_ondevice(model_id_for_bundle)
        elif PLUGIN == "qairt" and os.path.isdir(HOST_QAIRT_BUNDLES):
            adb(f"mkdir -p {DEVICE_QAIRT_BUNDLES}")
            subprocess.run(
                ["adb", "push", f"{HOST_QAIRT_BUNDLES}/.", DEVICE_QAIRT_BUNDLES],
                check=True,
            )
            names = [
                d
                for d in os.listdir(HOST_QAIRT_BUNDLES)
                if os.path.isdir(os.path.join(HOST_QAIRT_BUNDLES, d))
            ]
            assert len(names) == 1, f"expected one qairt bundle, got {names}"
            bundle_name = names[0]

        tsv_by_ctx: dict[int, list[str]] = {ctx: [] for ctx in CTXS}
        for row in rows:
            name, plugin, devs, model_id, vlm, _image = row.split("|")
            for d in devs.split(","):
                for ctx in CTXS:
                    tsv_by_ctx[ctx].append(
                        f"{name}-{plugin}-{d}-c{ctx}\t{plugin}\t{d}\t{model_id}"
                        f"\t\t\t\t{vlm}"
                    )
        assert any(tsv_by_ctx.values()), "no model rows produced"

        lib = f"{DEVICE_BUNDLE}/lib"
        env = (
            f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
            f"ADSP_LIBRARY_PATH={lib} "
            f"GENIEX_PLUGIN_PATH={lib}"
        )
        # Perf sweep records failures but defers the fatal check to the end so a
        # perf failure never skips the accuracy eval below.
        failures = []
        cell_json_count = 0
        results_listing = ""
        if RUN_PERF:
            for ctx in CTXS:
                tsv_path = f"/data/local/tmp/matrix-{ctx}.tsv"
                adb(
                    "printf '%s\\n' "
                    + " ".join(f"'{ln}'" for ln in tsv_by_ctx[ctx])
                    + f" > {tsv_path}"
                )
                if _run_bench(ctx, env, tsv_path, chipset, qairt_lib_flag) != 0:
                    failures.append(ctx)
            # Confirm cell JSONs exist; adb hides on-device exit codes.
            results_listing = adb(f"ls -l {DEVICE_RESULTS}", check=False).stdout
            count_proc = adb(f"ls {DEVICE_RESULTS} | wc -l", check=False)
            cell_json_count = (
                int(count_proc.stdout.strip().split()[-1])
                if count_proc.stdout.strip()
                else 0
            )
        else:
            print("perf sweep disabled (RUN_PERF=False); eval only")

        eval_ran = 0
        eval_collected = 0
        eval_expected = 0
        if RUN_EVAL:
            # Single model per job: derive the -m ref from the first row.
            _first_name, _, _first_devs, first_model = rows[0].split("|")[:4]
            eval_model = (
                f"{DEVICE_QAIRT_BUNDLES}/{bundle_name}"
                if PLUGIN == "qairt" and bundle_name
                else first_model
            )
            # Accuracy eval always runs on the NPU (the on-device target).
            eval_ran, eval_collected, eval_expected = _run_eval(
                env, eval_model, "npu", chipset, qairt_lib_flag
            )

        if RUN_PERF and (failures or cell_json_count == 0):
            # Surface geniex-bench's stderr so a device-side failure (e.g. a bad
            # QAIRT swap) is diagnosable from the JUnit XML alone.
            stderr_tail = adb(
                f"tail -n 40 {DEVICE_LOGS}/geniex_bench_stderr.log", check=False
            ).stdout
            pytest.fail(
                f"geniex-bench produced no usable output (failed ctxs={failures}, "
                f"cell_json_count={cell_json_count}).\n--- {DEVICE_RESULTS} ---\n"
                f"{results_listing}\n--- geniex_bench_stderr.log (tail) ---\n"
                f"{stderr_tail}"
            )
        # Every prompt failed => fail loudly so the host retry loop resubmits
        # instead of adb silently hiding the device drop.
        if RUN_EVAL and eval_ran == 0:
            pytest.fail("accuracy eval produced no output (all prompts failed)")
        # Prompts can all run and still leave nothing to fetch: a late adb drop
        # means device_logs is never harvested, yet the job reports "Successful", so
        # the host burns the run on eval_incomplete (0/N). Fail so the job is
        # reported as failed.
        if RUN_EVAL and eval_collected < eval_expected:
            pytest.fail(
                f"accuracy eval output incomplete on device: "
                f"{eval_collected}/{eval_expected} ===EVAL_IDX=== markers "
                f"readable back ({eval_ran} prompts ran). The device likely "
                f"dropped off adb mid-run, so these results would not be "
                f"retrievable by the host."
            )
    finally:
        _cleanup_device()


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    )
