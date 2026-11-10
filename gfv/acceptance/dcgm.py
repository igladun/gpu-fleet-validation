"""DCGM diagnostic (level 3 by default), plus per-GPU PCIe bandwidth from its pcie plugin."""

import json

from . import parse

TIMEOUT = {1: 600, 2: 1800, 3: 3600, 4: 6 * 3600}
# Tests a run at this level must report (DCGM 4.x names). Missing, or Skip on any GPU = error.
REQUIRED = {
    1: {"software"},
    2: {"software"},
    3: {"software", "memory", "diagnostic", "nvbandwidth", "pcie", "targeted_stress", "targeted_power"},
}
REQUIRED[4] = REQUIRED[3]


def run(ctx, level):
    ctx.stage("dcgm", f"dcgmi diag -r {level} (level 3 takes about 25 min)")
    if not ctx.which("dcgmi"):
        ctx.add("dcgm.diag", "error", None, None, None, "dcgmi not found")
        return
    ver = ctx.run("dcgmi-version", ["dcgmi", "--version"], timeout=30)
    version = parse.dcgmi_version(ver.out)
    ctx.rec.versions["dcgm"] = version
    if version and int(version.split(".")[0]) < 4:
        ctx.add("dcgm.diag", "error", None, version, ">= 4", "DCGM 4.x is required (JSON output format)")
        return
    ctx.facts["hostengine_started"] = False
    try:
        if ensure_hostengine(ctx) is not None:
            diag(ctx, level)
    finally:
        if ctx.facts["hostengine_started"]:
            stop_hostengine(ctx)


def ensure_hostengine(ctx):
    """True if we started nv-hostengine (and must stop it), False if one was running, None on failure.
    A started engine is always stopped by the caller, even if the probe after starting it fails."""
    if ctx.run("dcgmi-probe", ["dcgmi", "discovery", "-l"], timeout=60).ok:
        return False
    # Output goes to a file, not a pipe: a daemon that keeps the pipe open would hang the read.
    st = ctx.proc.run("nv-hostengine-start", ["nv-hostengine"], timeout=60, env_extra=ctx.env_extra(), to_file=True)
    if not st.ok:
        ctx.add("dcgm.diag", "error", None, None, None, f"no DCGM host engine and could not start one ({st.why()})")
        return None
    ctx.facts["hostengine_started"] = True          # from here on, run() always stops it
    ctx.notes.append("nv-hostengine was not running; started for the diagnostic and stopped afterwards")
    for _ in range(10):
        ctx.sleep(1)
        if ctx.run("dcgmi-probe-after-start", ["dcgmi", "discovery", "-l"], timeout=60).ok:
            return True
    ctx.add("dcgm.diag", "error", None, None, None, "started nv-hostengine but dcgmi cannot reach it")
    return None


def stop_hostengine(ctx):
    r = ctx.run("nv-hostengine-term", ["nv-hostengine", "--term"], timeout=60)
    if not r.ok:
        ctx.add("dcgm.hostengine_stop", "warn", None, None, None,
                f"could not stop the nv-hostengine this run started ({r.why()})")


def diag(ctx, level):
    d = ctx.run(f"dcgm-diag-r{level}", ["dcgmi", "diag", "-r", str(level), "-j"], timeout=TIMEOUT.get(level, 3600))
    try:
        rows, tests, meta = parse.dcgm_diag(d.out)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        ctx.add("dcgm.diag", "error", None, None, "Pass", f"no parseable JSON ({d.why()}; {e})", d.evidence)
        return
    if not rows and not tests:
        ctx.add("dcgm.diag", "error", None, None, "Pass", f"no test results ({d.why()})", d.evidence)
        return
    gpus = set(range(ctx.spec["gpus"]))
    required = REQUIRED.get(level, set())
    by_test = {}
    for r in rows:
        by_test.setdefault(r["test"], []).append(r)
    bad_total = 0
    for test in sorted(set(by_test) | set(tests) | required):
        rs = by_test.get(test, [])
        summary = (tests.get(test) or {}).get("summary")
        bad = 0
        if test not in by_test and test not in tests:
            ctx.add("dcgm.diag", "error", test, None, "Pass", f"required level-{level} test did not run", d.evidence)
            bad_total += 1
            continue
        for r in rs:
            verdict = parse.dcgm_verdict(r["status"])
            if verdict == "skip" and test in required:
                verdict = "error"
            if verdict != "pass":
                bad += 1
                ctx.add("dcgm.diag", verdict, f"{test}/{r['group']}{r['entity']}", r["status"], "Pass",
                        "; ".join(r["messages"])[:300], d.evidence)
        seen = {r["entity"] for r in rs if r["group"] == "GPU"}
        if rs and any(r["group"] == "GPU" for r in rs) and seen != gpus:
            bad += 1
            ctx.add("dcgm.diag", "error", test, sorted(seen), sorted(gpus), "test did not cover every GPU", d.evidence)
        if summary is not None and parse.dcgm_verdict(summary) in ("fail", "error"):
            bad += 1
            msgs = "; ".join((tests.get(test) or {}).get("messages", []))[:300]
            ctx.add("dcgm.diag", "fail", test, summary, "Pass", f"test summary: {msgs}", d.evidence)
        if not rs and test in required and not bad:
            bad += 1
            ctx.add("dcgm.diag", "error", test, summary, "Pass", "required test reported no per-GPU results", d.evidence)
        if not bad:
            ctx.add("dcgm.diag", "pass", test, f"{len(rs)}/{len(rs)} Pass", "Pass", evidence=d.evidence)
        bad_total += bad
    if not d.ok and not bad_total:
        ctx.add("dcgm.diag", "error", None, d.rc, 0, "dcgmi diag exited non-zero with no failing test in its JSON",
                d.evidence)
    pcie_floor(ctx, by_test.get("pcie", []), d.evidence)
    if meta:
        ctx.rec.versions.setdefault("dcgm", meta.get("version"))
        with open(f"{ctx.out_dir}/dcgm-metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)


def pcie_floor(ctx, rows, evidence):
    floor = ctx.perf("dcgm_pcie_bidir_gbs")
    if not rows:
        if floor is not None:
            ctx.add("dcgm.pcie_bidir", "error", None, None, floor, "no pcie plugin results at this level")
        return
    for r in sorted(rows, key=lambda r: (r["entity"] is None, r["entity"])):
        bw = parse.dcgm_pcie_bidir(r["info"])
        target = f"GPU{r['entity']}"
        if bw is None:
            ctx.add("dcgm.pcie_bidir", "error", target, None, floor, "no bidirectional bandwidth line", evidence)
        elif floor is None:
            ctx.add("dcgm.pcie_bidir", "info", target, bw, None, "GB/s; no floor in manifest", evidence)
        else:
            ctx.add("dcgm.pcie_bidir", "pass" if bw >= floor else "fail", target, bw, floor, "GB/s", evidence)
