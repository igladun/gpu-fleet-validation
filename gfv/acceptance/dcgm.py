"""DCGM diagnostic (level 3 by default), plus per-GPU PCIe bandwidth from its pcie plugin."""

import json

from . import parse

TIMEOUT = {1: 600, 2: 1800, 3: 3600, 4: 6 * 3600}


def run(ctx, level):
    ctx.stage("dcgm", f"dcgmi diag -r {level} (level 3 takes about 25 min)")
    if not ctx.which("dcgmi"):
        ctx.add("dcgm.diag", "error", None, None, None, "dcgmi not found")
        return
    ver = ctx.run("dcgmi-version", ["dcgmi", "--version"], timeout=30)
    ctx.rec.versions["dcgm"] = parse.dcgmi_version(ver.out)
    started = ensure_hostengine(ctx)
    if started is None:
        return
    try:
        diag(ctx, level)
    finally:
        if started:
            ctx.run("nv-hostengine-term", ["nv-hostengine", "--term"], timeout=60)


def ensure_hostengine(ctx):
    """True if we started nv-hostengine (and must stop it), False if one was running, None on failure."""
    if ctx.run("dcgmi-probe", ["dcgmi", "discovery", "-l"], timeout=60).ok:
        return False
    st = ctx.run("nv-hostengine-start", ["nv-hostengine"], timeout=60)
    ctx.sleep(3)
    if st.ok and ctx.run("dcgmi-probe-after-start", ["dcgmi", "discovery", "-l"], timeout=60).ok:
        ctx.notes.append("nv-hostengine was not running; started for the diagnostic and stopped afterwards")
        return True
    ctx.add("dcgm.diag", "error", None, None, None, f"no DCGM host engine and could not start one ({st.why()})")
    return None


def diag(ctx, level):
    d = ctx.run(f"dcgm-diag-r{level}", ["dcgmi", "diag", "-r", str(level), "-j"], timeout=TIMEOUT.get(level, 3600))
    try:
        rows, meta = parse.dcgm_diag(d.out)
    except (ValueError, KeyError, TypeError) as e:
        ctx.add("dcgm.diag", "error", None, None, "Pass", f"no parseable JSON ({d.why()}; {e})", d.evidence)
        return
    if not rows:
        ctx.add("dcgm.diag", "error", None, None, "Pass", f"no test results ({d.why()})", d.evidence)
        return
    tests = {}
    for r in rows:
        tests.setdefault(r["test"], []).append(r)
    for test, rs in tests.items():
        bad = [r for r in rs if parse.dcgm_verdict(r["status"]) != "pass"]
        for r in bad:
            ctx.add("dcgm.diag", parse.dcgm_verdict(r["status"]), f"{test}/{r['group']}{r['entity']}", r["status"],
                    "Pass", "; ".join(r["messages"])[:300], d.evidence)
        if not bad:
            ctx.add("dcgm.diag", "pass", test, f"{len(rs)}/{len(rs)} Pass", "Pass", evidence=d.evidence)
    pcie_floor(ctx, tests.get("pcie", []), d.evidence)
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
