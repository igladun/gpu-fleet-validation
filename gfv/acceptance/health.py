"""GPU, NVLink, NVSwitch and IB health: absolute state before the load, and what changed after it.

Missing or unparseable data is an error, never a pass."""

from . import parse
from .context import ib_devices, query_gpus

APP_XIDS = {13, 31, 43, 45}             # usually the application, not the hardware: warn; every other XID fails
IB_HARD = {"symbol_error", "link_downed", "link_error_recovery"}
IB_SOFT = {"port_rcv_errors", "port_rcv_remote_physical_errors", "port_xmit_discards",
           "local_link_integrity_errors", "excessive_buffer_overrun_errors"}
SATURATED = {"symbol_error": 65535, "link_error_recovery": 255, "link_downed": 255,
             "port_rcv_errors": 65535, "port_xmit_discards": 65535}
NVLINK_HARD = ("crc", "recovery")


def snapshot(ctx, phase):
    snap = {"gpus": query_gpus(ctx, f"health-gpus-{phase}")}
    rr = ctx.run(f"health-remapped-{phase}", ["nvidia-smi", "--query-remapped-rows=" + ",".join(parse.REMAP_FIELDS),
                                              "--format=csv,noheader"], timeout=60)
    try:
        snap["remap"] = parse.remapped_csv(rr.out) if rr.ok else None
    except ValueError:
        snap["remap"] = None
    dm = ctx.run(f"health-dmesg-{phase}", ["dmesg"], timeout=60)
    snap["xids"] = parse.xids(dm.out) if dm.ok else None
    snap["sxids"] = parse.sxids(dm.out) if dm.ok else None
    snap["dmesg_why"] = dm.why()
    snap["ib"] = ib_devices(ctx.sys)
    nv = ctx.run(f"health-nvlink-errors-{phase}", ["nvidia-smi", "nvlink", "-e"], timeout=60)
    snap["nvlink"] = parse.nvlink_errors(nv.out) if nv.ok else None
    return snap


def before(ctx):
    ctx.stage("health", "before load")
    s = snapshot(ctx, "before")
    ctx.facts["health_before"] = s
    if gpus_complete(ctx, s, "health.ecc_uncorrected"):
        bad = 0
        for g in s["gpus"]:
            vol = parse.as_int(g["ecc.errors.uncorrected.volatile.total"])
            agg = parse.as_int(g["ecc.errors.uncorrected.aggregate.total"])
            if vol is None or agg is None:
                bad += 1
                ctx.add("health.ecc_uncorrected", "error", f"GPU{g['index']}",
                        g["ecc.errors.uncorrected.volatile.total"], 0, "ECC counters unreadable (ECC disabled?)")
            elif vol:
                bad += 1
                ctx.add("health.ecc_uncorrected", "fail", f"GPU{g['index']}", vol, 0, "volatile, since driver load")
            elif agg:
                bad += 1
                ctx.add("health.ecc_uncorrected", "warn", f"GPU{g['index']}", agg, 0, "aggregate (lifetime)")
        if not bad:
            ctx.add("health.ecc_uncorrected", "pass", None, 0, 0)
    remap(ctx, s, "before")
    xid_check(ctx, s["xids"], s["dmesg_why"], "since boot")
    if ctx.spec["nvswitches"]:
        sxid_check(ctx, s["sxids"], "since boot", "warn")
    counters_present(ctx, s["ib"], "before")
    for dev, port, name, v in _ib_counters(ctx, s["ib"]):
        if name in SATURATED and v >= SATURATED[name]:
            ctx.add("health.ib_counter", "fail" if name in IB_HARD else "warn", f"{dev}/{port}", v, 0,
                    f"{name} is saturated: new errors can no longer be seen (reset the port counters)")
        elif v and name in IB_HARD | IB_SOFT:
            ctx.add("health.ib_counter", "warn", f"{dev}/{port}", v, 0, f"{name} non-zero since boot or reset")
    nvlink_complete(ctx, s["nvlink"], "before")


def after(ctx):
    ctx.stage("health", "after load")
    b = ctx.facts.get("health_before")
    s = snapshot(ctx, "after")
    by_idx = {g["index"]: g for g in (b or {}).get("gpus", [])}
    if gpus_complete(ctx, s, "health.ecc_delta"):
        bad = 0
        for g in s["gpus"]:
            was = parse.as_int(by_idx.get(g["index"], {}).get("ecc.errors.uncorrected.volatile.total"))
            now = parse.as_int(g["ecc.errors.uncorrected.volatile.total"])
            if was is None or now is None:
                bad += 1
                ctx.add("health.ecc_delta", "error", f"GPU{g['index']}", now, 0, "ECC counters unreadable")
            elif now > was:
                bad += 1
                ctx.add("health.ecc_delta", "fail", f"GPU{g['index']}", now - was, 0, "new uncorrectable ECC under load")
        if not bad:
            ctx.add("health.ecc_delta", "pass", None, 0, 0)
    remap(ctx, s, "after")
    if b is None or b["xids"] is None or s["xids"] is None:
        xid_check(ctx, s["xids"], s["dmesg_why"], "after load (no baseline)", check="health.xid_new")
    else:
        seen = {line for _, _, line in b["xids"]}
        xid_check(ctx, [x for x in s["xids"] if x[2] not in seen], s["dmesg_why"], "new during the run",
                  check="health.xid_new")
    if ctx.spec["nvswitches"]:
        if b is None or b["sxids"] is None or s["sxids"] is None:
            sxid_check(ctx, s["sxids"], "after load (no baseline)", "fail", check="health.sxid_new")
        else:
            seen = {line for _, _, line in b["sxids"]}
            sxid_check(ctx, [x for x in s["sxids"] if x[2] not in seen], "new during the run", "fail",
                       check="health.sxid_new")
    counters_present(ctx, s["ib"], "after")
    ib_delta(ctx, b["ib"] if b else None, s["ib"])
    nvlink_complete(ctx, s["nvlink"], "after")
    nvlink_delta(ctx, b["nvlink"] if b else None, s["nvlink"])


def gpus_complete(ctx, s, check):
    if len(s["gpus"]) != ctx.spec["gpus"]:
        ctx.add(check, "error", None, len(s["gpus"]), ctx.spec["gpus"], "GPU query failed or returned too few GPUs")
        return False
    return True


def remap(ctx, s, phase):
    rows = s["remap"]
    if rows is None or len(rows) != ctx.spec["gpus"]:
        ctx.add("health.remapped_rows", "error", None, None if rows is None else len(rows),
                "pending No, failure No", f"query failed or incomplete ({phase})")
        return
    bad = [r for r in rows if r["remapped_rows.pending"].lower() != "no" or r["remapped_rows.failure"].lower() != "no"]
    for r in bad:
        ctx.add("health.remapped_rows", "fail", r["bus"],
                f"pending {r['remapped_rows.pending']}, failure {r['remapped_rows.failure']}",
                "pending No, failure No", f"{phase}: drain, GPU reset, re-run; RMA on failure")
    if not bad:
        ctx.add("health.remapped_rows", "pass", None, "none pending or failed", "pending No, failure No", phase)


def xid_check(ctx, xids, why, scope, check="health.xid"):
    if xids is None:
        ctx.add(check, "error", None, None, "no XID", f"cannot read kernel log ({why})")
        return
    for bus, xid, line in xids:
        app = xid in APP_XIDS
        ctx.add(check, "warn" if app else "fail", bus, xid, None,
                f"{'application' if app else 'hardware or driver'} XID {scope}: {line[:160]}")
    if not xids:
        ctx.add(check, "pass", None, 0, 0, scope)


def sxid_check(ctx, sxids, scope, verdict, check="health.sxid"):
    if sxids is None:
        ctx.add(check, "error", None, None, "no SXid", "cannot read kernel log")
        return
    for bus, sxid, line in sxids:
        ctx.add(check, verdict, bus, sxid, None, f"NVSwitch SXid {scope}: {line[:160]}")
    if not sxids:
        ctx.add(check, "pass", None, 0, 0, scope)


def _ib_counters(ctx, ib):
    for dev in ctx.facts["rails"]:
        for port, p in sorted(ib.get(dev, {}).get("ports", {}).items()):
            for name, v in sorted(p["counters"].items()):
                yield dev, port, name, v


def counters_present(ctx, ib, phase):
    """Every rail must still exist and expose the hard error counters, or deltas would pass vacuously."""
    for dev in ctx.facts["rails"]:
        ports = ib.get(dev, {}).get("ports", {})
        missing = sorted(IB_HARD - set().union(*(set(p["counters"]) for p in ports.values()))) if ports else ["device"]
        if missing:
            ctx.add("health.ib_counters_readable", "error", dev, missing, sorted(IB_HARD),
                    f"error counters missing ({phase})")


def ib_delta(ctx, before_ib, after_ib):
    if before_ib is None:
        ctx.add("health.ib_counter_delta", "error", None, None, None, "no baseline")
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


def nvlink_complete(ctx, nv, phase):
    want = ctx.spec["nvlinks_total"]
    if not want:
        return
    links = {(gpu, link) for gpu, link, _ in (nv or {})}
    if len(links) != want:
        ctx.add("health.nvlink_counters_readable", "error", None, len(links), want,
                f"`nvidia-smi nvlink -e` did not report every link ({phase})")


def nvlink_delta(ctx, before_nv, after_nv):
    if not ctx.spec["nvlinks_total"]:
        return
    if not before_nv or not after_nv:
        ctx.add("health.nvlink_error_delta", "error", None, None, 0, "nvidia-smi nvlink -e gave no counters")
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
