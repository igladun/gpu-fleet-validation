"""Single-node NCCL all_reduce over NVLink/NVSwitch: bus bandwidth, correctness and NCCL version."""

import glob
import json
import os

from . import parse


def find_all_reduce(ctx):
    if ctx.nccl_tests:
        p = os.path.join(ctx.nccl_tests, "build", "all_reduce_perf")
        if os.access(p, os.X_OK):
            return p
    return ctx.which("all_reduce_perf")


MAX_BYTES = 8 * 1024 ** 3


def find_installed(tools_dir, nccl_want, cuda):
    """--no-install: a previous build for the manifest's NCCL and the driver's CUDA major, if any."""
    major = (cuda or "").split(".")[0]
    for meta_path in sorted(glob.glob(os.path.join(tools_dir, "nccl-tests-*", "gfv-tool.json")), reverse=True):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        v = meta.get("nccl_apt_version", "")
        if (nccl_want and not v.startswith(nccl_want + "-")) or f"+cuda{major}." not in v:
            continue
        if os.access(os.path.join(meta.get("dir", ""), "build", "all_reduce_perf"), os.X_OK):
            return meta
    return None


def run(ctx):
    n = ctx.spec["gpus"]
    exe = find_all_reduce(ctx)
    if not exe:
        ctx.add("nccl.all_reduce", "error", None, None, None, "all_reduce_perf not found")
        return
    ctx.stage("nccl", f"all_reduce_perf 8 B - 8 GiB on {n} GPUs")
    r = ctx.run("all_reduce_perf", [exe, "-b", "8", "-e", "8G", "-f", "2", "-g", str(n), "-n", "20", "-w", "5"],
                timeout=1800)
    log = parse.nccl_log(r.out)
    if not log["rows"]:
        ctx.add("nccl.all_reduce", "error", None, None, None, f"no result rows ({r.why()})", r.evidence)
        return
    ctx.rec.versions["nccl"] = log["nccl_version"]
    if log["ngpus"] != n:
        ctx.add("nccl.gpus", "fail", None, log["ngpus"], n, evidence=r.evidence)
    wrong = [row["size"] for row in log["rows"]
             if row["wrong"] not in ("0", "N/A") or row["wrong_ip"] not in ("0", "N/A")]
    ok = not wrong and log["oob_ok"] and r.ok
    detail = f"{len(log['rows'])} sizes" if ok else (
        f"wrong values at sizes {wrong[:5]}" if wrong else ("rc=%d" % r.rc if not r.ok else "out-of-bounds line missing"))
    ctx.add("nccl.correctness", "pass" if ok else "fail", None, "0 wrong" if ok else "errors", "0 wrong", detail,
            r.evidence)
    want = ctx.spec.get("nccl")
    if want:
        ctx.add("nccl.version", "pass" if log["nccl_version"] == want else "fail", None, log["nccl_version"], want,
                evidence=r.evidence)
    top = max(log["rows"], key=lambda row: row["size"])
    floor = ctx.perf("nccl_allreduce_busbw_gbs_1node")
    target = f"{top['size']} B"
    if top["size"] != MAX_BYTES:
        ctx.add("nccl.allreduce_busbw", "error", target, top["busbw"], floor,
                f"log ends before {MAX_BYTES} B (truncated run?)", r.evidence)
        return
    if floor is None:
        ctx.add("nccl.allreduce_busbw", "info", target, top["busbw"], None, "GB/s; no floor in manifest", r.evidence)
    else:
        ctx.add("nccl.allreduce_busbw", "pass" if (top["busbw"] or 0) >= floor else "fail", target, top["busbw"],
                floor, "GB/s out-of-place bus bandwidth at the largest size", r.evidence)
