"""GPU, NVLink and IB health: absolute state before the load, and what changed after it."""

from . import parse
from .context import ib_devices, query_gpus

HARD_XIDS = {48, 63, 64, 74, 79, 94, 95, 119, 120}     # hardware: drain, reset or RMA
APP_XIDS = {13, 31, 43}                                 # usually the application, not the hardware
IB_HARD = {"symbol_error", "link_downed", "link_error_recovery"}
IB_SOFT = {"port_rcv_errors", "port_rcv_remote_physical_errors", "port_xmit_discards",
           "local_link_integrity_errors", "excessive_buffer_overrun_errors"}
NVLINK_HARD = ("crc", "recovery")


def snapshot(ctx, phase):
    snap = {"gpus": query_gpus(ctx, f"health-gpus-{phase}")}
    rr = ctx.run(f"health-remapped-{phase}", ["nvidia-smi", "--query-remapped-rows=" + ",".join(parse.REMAP_FIELDS),
                                              "--format=csv,noheader"], timeout=60)
    snap["remap"] = parse.remapped_csv(rr.out) if rr.ok else None
    dm = ctx.run(f"health-dmesg-{phase}", ["dmesg"], timeout=60)
    snap["xids"] = parse.xids(dm.out) if dm.ok else None
    snap["dmesg_why"] = dm.why()
    snap["ib"] = ib_devices(ctx.sys)
    nv = ctx.run(f"health-nvlink-errors-{phase}", ["nvidia-smi", "nvlink", "-e"], timeout=60)
    snap["nvlink"] = parse.nvlink_errors(nv.out) if nv.ok else None
    return snap


def before(ctx):
    ctx.stage("health", "before load")
    s = snapshot(ctx, "before")
    ctx.facts["health_before"] = s
    for g in s["gpus"]:
        vol = parse.as_int(g["ecc.errors.uncorrected.volatile.total"])
        agg = parse.as_int(g["ecc.errors.uncorrected.aggregate.total"])
        if vol:
            ctx.add("health.ecc_uncorrected", "fail", f"GPU{g['index']}", vol, 0, "volatile, since driver load")
        elif agg:
            ctx.add("health.ecc_uncorrected", "warn", f"GPU{g['index']}", agg, 0, "aggregate (lifetime)")
    if s["gpus"] and not any(parse.as_int(g["ecc.errors.uncorrected.volatile.total"]) or
                             parse.as_int(g["ecc.errors.uncorrected.aggregate.total"]) for g in s["gpus"]):
        ctx.add("health.ecc_uncorrected", "pass", None, 0, 0)
    remap(ctx, s, "before")
    xid_check(ctx, s["xids"], s["dmesg_why"], "since boot")
    for dev, port, name, v in _ib_counters(ctx, s["ib"]):
        if v and name in IB_HARD | IB_SOFT:
            ctx.add("health.ib_counter", "warn", f"{dev}/{port}", v, 0, f"{name} non-zero since boot or reset")


def after(ctx):
    ctx.stage("health", "after load")
    b = ctx.facts.get("health_before")
    s = snapshot(ctx, "after")
    by_idx = {g["index"]: g for g in (b or {}).get("gpus", [])}
    bad = 0
    for g in s["gpus"]:
        was = parse.as_int(by_idx.get(g["index"], {}).get("ecc.errors.uncorrected.volatile.total")) or 0
        now = parse.as_int(g["ecc.errors.uncorrected.volatile.total"]) or 0
        if now > was:
            bad += 1
            ctx.add("health.ecc_delta", "fail", f"GPU{g['index']}", now - was, 0, "new uncorrectable ECC under load")
    if s["gpus"] and not bad:
        ctx.add("health.ecc_delta", "pass", None, 0, 0)
    remap(ctx, s, "after")
    if b is None or b["xids"] is None or s["xids"] is None:
        xid_check(ctx, s["xids"], s["dmesg_why"], "after load (no baseline)")
    else:
        seen = {line for _, _, line in b["xids"]}
        xid_check(ctx, [x for x in s["xids"] if x[2] not in seen], s["dmesg_why"], "new during the run",
                  check="health.xid_new")
    ib_delta(ctx, b["ib"] if b else None, s["ib"])
    nvlink_delta(ctx, b["nvlink"] if b else None, s["nvlink"])


def remap(ctx, s, phase):
    rows = s["remap"]
    if rows is None:
        ctx.add("health.remapped_rows", "error", None, None, "pending No, failure No",
                f"query failed ({phase})")
        return
    bad = [r for r in rows if r["remapped_rows.pending"].lower() != "no" or r["remapped_rows.failure"].lower() != "no"]
    for r in bad:
        ctx.add("health.remapped_rows", "fail", r["bus"],
                f"pending {r['remapped_rows.pending']}, failure {r['remapped_rows.failure']}",
                "pending No, failure No", f"{phase}: drain, GPU reset, re-run; RMA on failure")
    if rows and not bad:
        ctx.add("health.remapped_rows", "pass", None, "none pending or failed", "pending No, failure No", phase)


def xid_check(ctx, xids, why, scope, check="health.xid"):
    if xids is None:
        ctx.add(check, "error", None, None, "no hardware XID", f"cannot read kernel log ({why})")
        return
    for bus, xid, line in xids:
        verdict = "fail" if xid in HARD_XIDS else "warn"
        kind = "hardware" if xid in HARD_XIDS else ("application" if xid in APP_XIDS else "unclassified")
        ctx.add(check, verdict, bus, xid, None, f"{kind} XID {scope}: {line[:160]}")
    if not xids:
        ctx.add(check, "pass", None, 0, 0, scope)


def _ib_counters(ctx, ib):
    for dev in ctx.facts["rails"]:
        for port, p in sorted(ib.get(dev, {}).get("ports", {}).items()):
            for name, v in sorted(p["counters"].items()):
                yield dev, port, name, v


def ib_delta(ctx, before_ib, after_ib):
    if before_ib is None:
        ctx.add("health.ib_counter_delta", "skip", None, None, None, "no baseline")
        return
    base = {(d, p, n): v for d, p, n, v in _ib_counters(ctx, before_ib)}
    bad = 0
    for dev, port, name, v in _ib_counters(ctx, after_ib):
        if name not in IB_HARD | IB_SOFT:
            continue
        diff = v - base.get((dev, port, name), 0)
        if diff > 0:
            bad += 1
            ctx.add("health.ib_counter_delta", "fail" if name in IB_HARD else "warn", f"{dev}/{port}", diff, 0,
                    f"{name} increased under load")
    if not bad:
        ctx.add("health.ib_counter_delta", "pass", None, 0, 0, f"{len(ctx.facts['rails'])} rails")


def nvlink_delta(ctx, before_nv, after_nv):
    if before_nv is None or after_nv is None:
        if ctx.spec["nvlinks_total"]:
            ctx.add("health.nvlink_error_delta", "error", None, None, 0, "nvidia-smi nvlink -e failed")
        return
    bad = 0
    for key, v in sorted(after_nv.items()):
        diff = v - before_nv.get(key, 0)
        if diff > 0:
            bad += 1
            gpu, link, name = key
            hard = any(h in name.lower() for h in NVLINK_HARD)
            ctx.add("health.nvlink_error_delta", "fail" if hard else "warn", f"GPU{gpu}/link{link}", diff, 0,
                    f"{name} increased under load")
    if not bad:
        ctx.add("health.nvlink_error_delta", "pass", None, 0, 0)
