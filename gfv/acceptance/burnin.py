"""Burn-in: every GPU held at its power limit for N minutes (DCGM targeted_power), with temperature, power and
clock telemetry every 5 s, and the throttle counters read before and after.

Power is heat, so a full-power soak is the worst thermal case a GPU sees. The checks: the GPU's own temperature
limits match the manifest, the load really covered the requested time, the hottest GPU and HBM readings stay under
the burn-in thresholds, no GPU runs much hotter than the others, no thermal or power-brake throttling, the power
held, and temperatures levelled off. Missing data where the manifest sets a threshold is an error, never a skip."""

import os
import re
import statistics

from . import dcgm, parse

# DCGM field IDs: driver-independent, unlike nvidia-smi query names and its -q temperature layout.
LIMITS = (("gpu_max_operating_c", 152), ("gpu_slowdown_c", 158), ("gpu_shutdown_c", 159),
          ("mem_max_operating_c", 151))
GPU_TEMP, MEM_TEMP, POWER, SM_CLOCK = 150, 140, 155, 100
TELEMETRY = (GPU_TEMP, MEM_TEMP, POWER, SM_CLOCK)
INTERVAL_S = 5
THROTTLE = ("clocks_event_reasons_counters.sw_thermal_slowdown",
            "clocks_event_reasons_counters.hw_thermal_slowdown",
            "clocks_event_reasons_counters.hw_power_brake_slowdown")
THROTTLE_NAMES = ("SW thermal slowdown", "HW thermal slowdown", "HW power brake")
STEADY_WINDOW_S = 300          # the last 5 loaded minutes decide steady state and the per-GPU steady temperature
STEADY_SLOPE = 0.5             # degrees C per minute still rising = not levelled off
MIN_STEADY_S = 600             # shorter runs cannot judge steady state
MIN_COVERAGE = 0.9             # loaded samples must cover this share of the requested time
MIN_REPORTED = 0.9             # a thresholded value must be reported in this share of the samples


def run(ctx, minutes=None):
    if "burnin" not in ctx.spec:
        if minutes:
            ctx.add("burnin", "error", None, None, None,
                    "--burnin-minutes was given but the manifest has no [burnin] table with thresholds")
        else:
            ctx.add("burnin", "skip", None, None, None, "no [burnin] table in the manifest")
        return
    bt = ctx.table("burnin")
    minutes = minutes or bt.get("minutes", 30)
    ctx.stage("burnin", f"{minutes} min at the power limit (dcgmi diag -r targeted_power), "
                        f"telemetry every {INTERVAL_S} s")
    if not ctx.which("dcgmi"):
        ctx.add("burnin", "error", None, None, None, "dcgmi not found")
        return
    ctx.facts["hostengine_started"] = False
    try:
        if dcgm.ensure_hostengine(ctx, check="burnin") is not None:
            limits(ctx)
            soak(ctx, minutes, bt)
    finally:
        if ctx.facts["hostengine_started"]:
            dcgm.stop_hostengine(ctx)


def limits(ctx):
    """The GPU's own temperature limits, compared with [thermal]. Every GPU must report each limit the
    manifest sets."""
    fields = [f for _, f in LIMITS]
    r = ctx.run("burnin-limits", ["dcgmi", "dmon", "-e", ",".join(map(str, fields)), "-c", "1"], timeout=60)
    data = parse.dcgmi_dmon(r.out, fields) if r.ok else {}
    want_all = ctx.table("thermal")
    gpus = range(ctx.spec["gpus"])
    found = {}
    for key, field in LIMITS:
        known = {g: data[g][0][field] for g in gpus if data.get(g) and data[g][0][field] is not None}
        missing = [g for g in gpus if g not in known]
        want = want_all.get(key)
        if not known or (missing and want is not None):
            why = r.why() if not r.ok else (f"not reported for GPUs {missing}" if known else "not reported by DCGM")
            ctx.add("burnin.limits", "error" if want is not None or not r.ok else "skip", key, None, want, why,
                    r.evidence)
            continue
        distinct = sorted({_n(v) for v in known.values()})
        if len(distinct) == 1:
            found[key] = distinct[0]
        if want is None:
            ctx.add("burnin.limits", "info", key, distinct, None, "°C; not in the manifest", r.evidence)
            continue
        bad = {g: v for g, v in known.items() if v != want}
        for g, v in sorted(bad.items()):
            ctx.add("burnin.limits", "fail", f"{key} GPU{g}", _n(v), want, "°C; another SKU or VBIOS?", r.evidence)
        if not bad:
            ctx.add("burnin.limits", "pass", key, want, want, f"°C on {len(known)} GPUs", r.evidence)
    ctx.facts["thermal_limits"] = found


def soak(ctx, minutes, bt):
    secs = int(round(minutes * 60))
    before = throttle_counters(ctx, "before")
    # -c bounds dmon by itself (the load plus the diag's 900 s timeout margin), so it cannot outlive a killed run.
    count = secs // INTERVAL_S + 900 // INTERVAL_S
    mon = ctx.spawn("burnin-dmon", ["dcgmi", "dmon", "-e", ",".join(map(str, TELEMETRY)),
                                    "-d", str(INTERVAL_S * 1000), "-c", str(count)])
    early = False
    try:
        d = ctx.run("burnin-targeted-power", ["dcgmi", "diag", "-r", "targeted_power", "-p",
                                              f"targeted_power.test_duration={secs}", "-j"], timeout=secs + 900)
    finally:
        if mon is not None:
            early = mon.p.poll() is not None
            mon.stop()
    after = throttle_counters(ctx, "after")
    dcgm_result(ctx, d)
    if mon is None:
        ctx.add("burnin.telemetry", "error", None, None, None, "dcgmi dmon could not be started")
    else:
        ev = f"raw/{mon.name}.out"
        with open(os.path.join(ctx.out_dir, ev), errors="replace") as f:
            samples = parse.dcgmi_dmon(f.read(), TELEMETRY)
        note = "; dcgmi dmon exited before the load ended" if early else ""
        telemetry(ctx, samples, bt, secs, ev, note)
    throttle(ctx, before, after)


def dcgm_result(ctx, d):
    try:
        rows, tests, _ = parse.dcgm_diag(d.out)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        ctx.add("burnin.dcgm", "error", None, None, "Pass", f"no parseable JSON ({d.why()}; {e})", d.evidence)
        return
    rs = [r for r in rows if r["test"] == "targeted_power" and r["group"] == "GPU"]
    summary = (tests.get("targeted_power") or {}).get("summary")
    if not rs:
        ctx.add("burnin.dcgm", "error", None, summary, "Pass",
                f"targeted_power reported no per-GPU results ({d.why()})", d.evidence)
        return
    bad = 0
    for r in rs:
        verdict = parse.dcgm_verdict(r["status"])
        if verdict != "pass":
            bad += 1
            ctx.add("burnin.dcgm", "error" if verdict == "skip" else verdict, f"GPU{r['entity']}", r["status"],
                    "Pass", "; ".join(r["messages"] + list(r["info"]))[:300], d.evidence)
    seen = {r["entity"] for r in rs}
    if seen != set(range(ctx.spec["gpus"])):
        bad += 1
        ctx.add("burnin.dcgm", "error", None, sorted(seen), list(range(ctx.spec["gpus"])),
                "targeted_power did not cover every GPU", d.evidence)
    if not bad and summary is not None and parse.dcgm_verdict(summary) in ("fail", "error"):
        bad += 1
        ctx.add("burnin.dcgm", "fail", None, summary, "Pass", "targeted_power test summary", d.evidence)
    if not bad and not d.ok:
        bad += 1
        ctx.add("burnin.dcgm", "error", None, d.rc, 0, "dcgmi diag exited non-zero with no failing GPU", d.evidence)
    if not bad:
        avg = [float(m) for r in rs for line in r["info"]
               for m in re.findall(r"average power usage:\s*([0-9.]+)\s*W", line)]
        power = f"; DCGM average power {min(avg):g}-{max(avg):g} W" if avg else ""
        ctx.add("burnin.dcgm", "pass", None, f"{len(rs)}/{len(rs)} Pass", "Pass",
                f"targeted_power for {d.seconds:.0f} s{power}", d.evidence)


def loaded_threshold(ctx, bt, s):
    """Power above which a sample counts as 'under load'."""
    if ctx.spec.get("gpu_power_limit_w"):
        return 0.8 * ctx.spec["gpu_power_limit_w"]
    if bt.get("power_min_w"):
        return 0.85 * bt["power_min_w"]
    watts = [x[POWER] for x in s if x[POWER] is not None]
    return 0.8 * max(watts) if watts else None


def telemetry(ctx, samples, bt, secs, ev, note=""):
    gpus = list(range(ctx.spec["gpus"]))
    if not any(samples.get(g) for g in gpus):
        ctx.add("burnin.telemetry", "error", None, None, None, "no dcgmi dmon samples" + note, ev)
        return
    write_curve(ctx, samples)
    limits_found = ctx.facts.get("thermal_limits") or {}
    thermal = ctx.table("thermal")
    gpu_limit = limits_found.get("gpu_max_operating_c", thermal.get("gpu_max_operating_c"))
    mem_limit = limits_found.get("mem_max_operating_c", thermal.get("mem_max_operating_c"))
    steady_n = max(1, STEADY_WINDOW_S // INTERVAL_S)
    steady, slopes, short = {}, {}, False
    for g in gpus:
        s = samples.get(g) or []
        target = f"GPU{g}"
        if not s:
            ctx.add("burnin.telemetry", "error", target, 0, secs, "GPU missing from the telemetry" + note, ev)
            continue
        thresh = loaded_threshold(ctx, bt, s)
        loaded = [x for x in s if thresh is not None and x[POWER] is not None and x[POWER] >= thresh]
        loaded_ids = {id(x) for x in loaded}
        covered = len(loaded) * INTERVAL_S
        if covered < MIN_COVERAGE * secs:
            short = True
            ctx.add("burnin.telemetry", "error", target, covered, secs,
                    f"s under load in the telemetry, of {secs} s requested: the load or the monitor stopped early"
                    + note, ev)
        else:
            ctx.add("burnin.telemetry", "pass", target, covered, secs, f"s under load, sampled every {INTERVAL_S} s",
                    ev)
        temp_check(ctx, "burnin.gpu_temp", g, [x[GPU_TEMP] for x in s], bt.get("gpu_temp_max_c"),
                   bt.get("gpu_temp_warn_c"), gpu_limit, "GPU", ev)
        temp_check(ctx, "burnin.mem_temp", g, [x[MEM_TEMP] for x in s], bt.get("mem_temp_max_c"), None,
                   mem_limit, "HBM", ev)
        power_check(ctx, g, loaded or s, bt.get("power_min_w"), ev)
        # Steady state and spread from loaded samples only: the cool-down after the load would bias both.
        tail = [(i * INTERVAL_S / 60, x[GPU_TEMP]) for i, x in enumerate(s)
                if id(x) in loaded_ids and x[GPU_TEMP] is not None][-steady_n:]
        if tail:
            steady[g] = statistics.median(t for _, t in tail)
        if len(tail) >= 2 and covered >= MIN_STEADY_S:
            slopes[g] = statistics.linear_regression([m for m, _ in tail], [t for _, t in tail]).slope
    steady_state(ctx, secs, slopes, short, ev)
    spread(ctx, steady, bt.get("gpu_temp_spread_c"), ev)


def temp_check(ctx, check, g, values, fail_c, warn_c, limit, what, ev):
    vals = [v for v in values if v is not None]
    target = f"GPU{g}"
    gated = fail_c is not None or warn_c is not None
    if not vals or len(vals) < MIN_REPORTED * len(values):
        why = f"{what} temperature reported in {len(vals)} of {len(values)} samples"
        ctx.add(check, "error" if gated else "skip", target, None, fail_c, why, ev)
        return
    peak = _n(max(vals))
    margin = f"; {_n(limit - peak)} °C under the {_n(limit)} °C max operating temperature" if limit else ""
    if fail_c is not None and peak > fail_c:
        verdict = "fail"
    elif warn_c is not None and peak > warn_c:
        verdict = "warn"
    else:
        verdict = "pass" if gated else "info"
    ctx.add(check, verdict, target, peak, fail_c, f"°C peak{margin}", ev)


def power_check(ctx, g, s, want, ev):
    window = s[len(s) // 3:] or s           # the first third is ramp-up
    watts = [x[POWER] for x in window if x[POWER] is not None]
    clocks = [x[SM_CLOCK] for x in window if x[SM_CLOCK] is not None]
    target = f"GPU{g}"
    if not watts or len(watts) < MIN_REPORTED * len(window):
        why = f"power reported in {len(watts)} of {len(window)} samples"
        ctx.add("burnin.power", "error" if want is not None else "skip", target, None, want, why, ev)
        return
    med = round(statistics.median(watts), 1)
    clock = f"; SM clock median {_n(statistics.median(clocks))} MHz" if clocks else ""
    detail = f"W median under load ({len(window) * INTERVAL_S / 60:.1f} min); peak {_n(round(max(watts), 1))} W{clock}"
    if want is None:
        ctx.add("burnin.power", "info", target, med, None, detail, ev)
    else:
        ctx.add("burnin.power", "pass" if med >= want else "fail", target, med, want, detail, ev)


def steady_state(ctx, secs, slopes, short, ev):
    if secs < MIN_STEADY_S:
        ctx.add("burnin.steady_state", "info", None, None, None,
                f"run shorter than {MIN_STEADY_S // 60} min; steady state not judged", ev)
        return
    if short or not slopes:                     # burnin.telemetry already records the short run as an error
        ctx.add("burnin.steady_state", "skip", None, None, f"<= {STEADY_SLOPE} °C/min",
                "not enough loaded samples to judge steady state (see burnin.telemetry)", ev)
        return
    rising = {f"GPU{g}": round(v, 2) for g, v in sorted(slopes.items()) if v > STEADY_SLOPE}
    worst = round(max(slopes.values()), 2)
    if rising:
        ctx.add("burnin.steady_state", "warn", None, rising, f"<= {STEADY_SLOPE} °C/min",
                "temperature still rising over the last 5 min; run a longer burn-in (--burnin-minutes)", ev)
    else:
        ctx.add("burnin.steady_state", "pass", None, worst, f"<= {STEADY_SLOPE} °C/min",
                "°C/min, steepest GPU over the last 5 min under load", ev)


def spread(ctx, steady, limit, ev):
    if len(steady) < 2:
        return
    hot = max(steady, key=steady.get)
    cold = min(steady, key=steady.get)
    value = _n(round(steady[hot] - steady[cold], 1))
    detail = (f"°C; hottest GPU{hot} {_n(steady[hot])} °C, coolest GPU{cold} {_n(steady[cold])} °C "
              "(median of the last 5 min under load); a large gap points at a fan or heatsink zone")
    if limit is None:
        ctx.add("burnin.temp_spread", "info", None, value, None, detail, ev)
    else:
        ctx.add("burnin.temp_spread", "warn" if value > limit else "pass", None, value, limit, detail, ev)


def throttle_counters(ctx, phase):
    q = ctx.run(f"burnin-throttle-{phase}", ["nvidia-smi", "--query-gpu=index," + ",".join(THROTTLE),
                                             "--format=csv,noheader,nounits"], timeout=60)
    return (parse.gpu_values(q.out) if q.ok else None), q


def throttle(ctx, before, after):
    """Fail closed: unreadable, unparseable or N/A counters are an error, because a throttle would go unseen."""
    (b, bq), (a, aq) = before, after
    if not b or not a:
        q = bq if not b else aq
        why = q.why() if not q.ok else "no counter rows parsed"
        ctx.add("burnin.throttle", "error", None, None, 0, f"throttle counters unreadable ({why})", q.evidence)
        return
    bad = 0
    for g in range(ctx.spec["gpus"]):
        vb, va = b.get(g), a.get(g)
        if not vb or not va or len(vb) != len(THROTTLE) or len(va) != len(THROTTLE) or None in vb + va:
            bad += 1
            ctx.add("burnin.throttle", "error", f"GPU{g}", None, 0, "throttle counters unreadable or N/A",
                    aq.evidence)
            continue
        hit = {name: round((y - x) / 1e6, 3) for name, x, y in zip(THROTTLE_NAMES, vb, va) if y > x}
        if hit:
            bad += 1
            ctx.add("burnin.throttle", "fail", f"GPU{g}", hit, 0,
                    "seconds of clock throttling during the burn-in (thermal or platform power brake)", aq.evidence)
    if not bad:
        ctx.add("burnin.throttle", "pass", None, 0, 0, "no thermal slowdown or power brake during the burn-in",
                aq.evidence)


def write_curve(ctx, samples):
    """burnin-temperatures.csv: seconds, then one GPU temperature column per GPU, for plotting."""
    gpus = sorted(samples)
    n = max(len(samples[g]) for g in gpus)
    with open(os.path.join(ctx.out_dir, "burnin-temperatures.csv"), "w") as f:
        f.write("seconds," + ",".join(f"gpu{g}_c" for g in gpus) + "\n")
        for i in range(n):
            row = [samples[g][i][GPU_TEMP] if i < len(samples[g]) else None for g in gpus]
            f.write(f"{i * INTERVAL_S}," + ",".join("" if v is None else f"{_n(v)}" for v in row) + "\n")


def _n(v):
    """87.0 -> 87 for display; other floats unchanged."""
    return int(v) if isinstance(v, float) and v.is_integer() else v
