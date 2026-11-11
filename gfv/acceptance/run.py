"""L2 acceptance runner: stage order, fail-closed verdict, evidence directory and exit codes."""

import json
import os
import secrets
import signal
import socket
import sys
import traceback

from .. import build_info
from ..common import install, manifest
from ..common.proc import Proc
from ..common.results import Recorder, utcnow
from ..common.sysinfo import SysFS
from . import burnin, dcgm, fio, health, inventory, nccl, nvlink, parse, rdma
from .context import Ctx, discover

STAGES = ("install", "inventory", "nvlink", "rdma", "health", "dcgm", "burnin", "nccl", "fio")
DEFAULT_OUT = "/var/lib/gpu-fleet-validation/runs"
DEFAULT_TOOLS = "/opt/gpu-fleet-validation/tools"
EXIT_PASS, EXIT_FAIL, EXIT_CANNOT_RUN = 0, 1, 2


def add_arguments(p):
    p.add_argument("--class", dest="cls", help="node class (bundled manifest, or a class in --manifest)")
    p.add_argument("--manifest", help="manifest file (.toml needs Python 3.11+, .json works everywhere)")
    p.add_argument("--tools-dir", default=DEFAULT_TOOLS, help=f"where test tools are built (default {DEFAULT_TOOLS})")
    p.add_argument("--no-install", action="store_true", help="never install anything; missing tools fail their checks")
    p.add_argument("--out", help=f"evidence directory (default {DEFAULT_OUT}/<run_id>)")
    p.add_argument("--only", help="comma-separated stages to run: " + ",".join(STAGES))
    p.add_argument("--skip", help="comma-separated stages to skip")
    p.add_argument("--dcgm-level", type=int, default=3, choices=(1, 2, 3, 4), help="dcgmi diag level (default 3)")
    p.add_argument("--burnin-minutes", type=_positive_int, metavar="N",
                   help="burn-in length at the power limit (default: the manifest's burnin.minutes)")
    p.add_argument("--node", default=socket.gethostname(), help="node name in results (default: hostname)")
    p.add_argument("--cluster", help="cluster label for results")
    p.add_argument("--rack", help="rack label for results")
    p.add_argument("--leaf", help="leaf-switch label for results")
    p.add_argument("--force", action="store_true", help="run even if GPU processes are running")
    p.add_argument("--fail-fast", action="store_true", help="skip the heavy stages once a cheap check failed")
    p.add_argument("--allow-partial", action="store_true",
                   help="with --only/--skip: let a partial run pass (it fails by default; e.g. pre-staging tools)")
    p.add_argument("--dry-run", action="store_true", help="validate the manifest and print the plan; run nothing")


def _positive_int(v):
    n = int(v)
    if n < 1:
        raise ValueError(v)
    return n


def stages_from(args):
    def parse_list(v):
        names = [s.strip() for s in (v or "").split(",") if s.strip()]
        bad = [n for n in names if n not in STAGES]
        if bad:
            raise ValueError(f"unknown stage(s) {bad}; known: {', '.join(STAGES)}")
        return names
    only, skip = parse_list(args.only), parse_list(args.skip)
    return [s for s in STAGES if (not only or s in only) and s not in skip]


class Terminated(BaseException):
    """SIGTERM/SIGHUP: unwind through every finally (stop servers, the host engine, remove fio data)."""


def _on_signal(signum, _frame):
    for sig in (signal.SIGTERM, signal.SIGHUP):      # a second signal must not abort the cleanup
        signal.signal(sig, signal.SIG_IGN)
    raise Terminated(signal.Signals(signum).name)


TEST_OVERRIDES = ("GFV_SYSROOT", "GFV_ALLOW_NONROOT", "GFV_SETTLE_SECONDS")


def accept(args):
    try:
        stages = stages_from(args)
        cls, spec, source, _ = manifest.load(args.manifest, args.cls)
    except Exception as e:  # noqa: BLE001 - any bad manifest or argument means the gate cannot run
        print(f"gfv: {e}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if args.no_install:
        stages = [s for s in stages if s != "install"]
    if not stages:
        print("gfv: no stages selected", file=sys.stderr)
        return EXIT_CANNOT_RUN
    missing = [s for s in STAGES if s not in stages and not (s == "install" and args.no_install)]
    if args.dry_run:
        print_plan(cls, spec, source, stages, args)
        if missing:
            print(f"partial run: {','.join(missing)} not selected" +
                  ("" if args.allow_partial else " (exit 1 unless --allow-partial)"))
        return EXIT_PASS
    if os.geteuid() != 0 and not os.environ.get("GFV_ALLOW_NONROOT"):
        print("gfv: run as root (DCGM diagnostics, kernel log, PCI config space, tool install)", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if args.out and os.path.isdir(args.out) and os.listdir(args.out):
        print(f"gfv: evidence directory {args.out} is not empty", file=sys.stderr)
        return EXIT_CANNOT_RUN

    run_id = f"l2-{utcnow().replace('-', '').replace(':', '')}-{secrets.token_hex(2)}"
    out_dir = args.out or os.path.join(DEFAULT_OUT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"class": cls, "source": source, "spec": spec}, f, indent=2)
    context = {"run_id": run_id, "cluster": args.cluster, "node": args.node, "class": cls, "rack": args.rack,
               "leaf": args.leaf, "pod": None, "tier": "L2"}
    rec = Recorder(context, out_dir)
    rec.versions["gfv"] = build_info()["label"]
    ctx = Ctx(args, cls, spec, out_dir, Proc(out_dir), SysFS(os.environ.get("GFV_SYSROOT", "/")), rec,
              settle=float(os.environ.get("GFV_SETTLE_SECONDS", "1.5")))
    print(f"gfv L2 acceptance {run_id}: node {args.node}, class {cls} ({source}), stages {','.join(stages)}",
          file=sys.stderr)
    handled = [sig for sig in (signal.SIGTERM, signal.SIGHUP)
               if signal.getsignal(sig) is not signal.SIG_IGN]      # nohup: keep ignoring SIGHUP
    for sig in handled:
        signal.signal(sig, _on_signal)
    rec.add("run.stages", "info", None, stages, list(STAGES))
    overrides = sorted(k for k in TEST_OVERRIDES if k in os.environ)
    if overrides:
        rec.add("run.test_overrides", "warn", None, overrides, None,
                "test-only environment overrides are set; results may not describe this host")
    for s in missing:
        rec.add(f"{s}.not_selected", "skip", None, None, None, "stage not selected")
    if missing:
        rec.add("run.partial", "warn" if args.allow_partial else "error", None, missing, [],
                "partial run: these stages did not run" + ("" if args.allow_partial else
                                                            " (pass --allow-partial to accept a partial run)"))
    code = EXIT_FAIL
    try:
        code = run_stages(ctx, stages)
    except (KeyboardInterrupt, Terminated) as e:
        rec.add("run", "error", None, None, None, f"interrupted ({type(e).__name__} {e}); the gate did not finish")
        code = EXIT_CANNOT_RUN
    except Exception as e:  # noqa: BLE001 - a crash outside a stage still leaves a FAIL record
        path = os.path.join(out_dir, "crash-run.txt")
        with open(path, "w") as f:
            f.write(traceback.format_exc())
        rec.add("run.crash", "error", None, None, None, f"{type(e).__name__}: {e}", os.path.basename(path))
        prereq_ok = any(r["check"] == "prereq.gpus_idle" and r["verdict"] in ("pass", "warn") for r in rec.results)
        code = EXIT_FAIL if prereq_ok else EXIT_CANNOT_RUN
    finally:
        for sig in handled:
            signal.signal(sig, signal.SIG_DFL)
        if code == EXIT_PASS and rec.overall() != "pass":
            code = EXIT_FAIL
        rec.finish({"notes": ctx.notes, "finished": utcnow()})
        verdict = "PASS" if code == EXIT_PASS else ("CANNOT RUN" if code == EXIT_CANNOT_RUN else "FAIL")
        if missing and code == EXIT_PASS:
            verdict = f"PASS (partial: {','.join(missing)} not run)"
        print(f"gfv: {verdict} - evidence in {out_dir} (SUMMARY.md, results.jsonl, junit.xml)", file=sys.stderr)
    return code


def run_stages(ctx, stages):
    rec, args = ctx.rec, ctx.args
    ctx.stage("prereq")
    discover(ctx)
    if not ctx.facts["smi_ok"]:
        rec.add("prereq.nvidia_smi", "error", None, None, "working driver", ctx.facts["smi_why"],
                "raw/nvidia-smi.out")
        return EXIT_CANNOT_RUN
    rec.add("prereq.nvidia_smi", "pass", None, ctx.facts.get("driver"), None, f"{len(ctx.facts['gpus'])} GPUs")
    apps = ctx.run("nvidia-smi-compute-apps", ["nvidia-smi", "--query-compute-apps=" + ",".join(parse.APP_FIELDS),
                                               "--format=csv,noheader"], timeout=60)
    if not apps.ok:
        rec.add("prereq.gpus_idle", "error", None, None, 0, f"cannot list GPU processes ({apps.why()})",
                apps.evidence)
        return EXIT_CANNOT_RUN
    busy = parse.compute_apps(apps.out)
    if busy:
        names = sorted({a.get("process_name", "?") for a in busy})
        if not args.force:
            rec.add("prereq.gpus_idle", "error", None, len(busy), 0,
                    f"GPU processes running ({', '.join(names)[:200]}); stop them or pass --force", apps.evidence)
            return EXIT_CANNOT_RUN
        rec.add("prereq.gpus_idle", "warn", None, len(busy), 0, f"--force with processes running: {names}")
    else:
        rec.add("prereq.gpus_idle", "pass", None, 0, 0)

    if "install" in stages:
        # `--only install` pre-stages a node (e.g. for an image bake): install everything.
        _guard(ctx, "install", lambda: install.run(ctx, args.tools_dir,
                                                   nccl_needed="nccl" in stages or stages == ["install"]))
    else:
        meta = nccl.find_installed(args.tools_dir, ctx.spec.get("nccl"), ctx.facts.get("cuda"))
        if meta:
            install.use_tools(ctx, meta)

    if "health" in stages:
        _guard(ctx, "health", lambda: health.before(ctx))
    if "inventory" in stages:
        _guard(ctx, "inventory", lambda: inventory.run(ctx))
    if "nvlink" in stages:
        _guard(ctx, "nvlink", lambda: nvlink.run(ctx))
    if "rdma" in stages:
        _guard(ctx, "rdma", lambda: rdma.run_ports(ctx))

    blocked = args.fail_fast and bool(rec.failing())
    heavy = [("rdma", lambda: rdma.run_bandwidth(ctx)), ("dcgm", lambda: dcgm.run(ctx, args.dcgm_level)),
             ("burnin", lambda: burnin.run(ctx, args.burnin_minutes)),
             ("nccl", lambda: nccl.run(ctx)), ("fio", lambda: fio.run(ctx))]
    for name, fn in heavy:
        if name not in stages:
            continue
        if blocked:
            rec.add(f"{name}.load", "skip", None, None, None, "--fail-fast: an earlier check failed")
            continue
        _guard(ctx, name, fn)
    if "health" in stages:
        _guard(ctx, "health", lambda: health.after(ctx))
    return EXIT_PASS if rec.overall() == "pass" else EXIT_FAIL


def _guard(ctx, stage, fn):
    """A bug in one stage is an error result, not a crashed run (fail-closed, evidence kept)."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - every failure must end up in the results
        path = os.path.join(ctx.out_dir, f"crash-{stage}.txt")
        with open(path, "a") as f:
            f.write(traceback.format_exc())
        ctx.add(f"{stage}.crash", "error", None, None, None, f"{type(e).__name__}: {e}", os.path.basename(path))


PLAN = {
    "install": ["apt-get install (missing only): fio pciutils mokutil git build-essential perftest",
                "apt-get install datacenter-gpu-manager-4-cuda<driver CUDA major> (if dcgmi missing)",
                "apt-get download libnccl2/libnccl-dev=<nccl>+cuda<driver CUDA> ; dpkg-deb -x into <tools-dir>",
                "git fetch nccl-tests@<pinned commit> ; make NCCL_HOME=<tools-dir>/nccl-..."],
    "inventory": ["nvidia-smi --query-gpu=...", "nvidia-smi --query-gpu=index,enforced.power.limit",
                  "/proc/cmdline, /sys/devices/system/node", "mokutil --sb-state",
                  "lspci -D -vvv (ACS on bridges above GPUs)", "/proc/modules (nvidia_peermem)", "ofed_info -s",
                  "/sys/class/dmi/id/bios_version, /sys/class/infiniband/*/fw_ver"],
    "nvlink": ["nvidia-smi nvlink -s", "nvidia-smi -q (Fabric state)", "lspci -nn -d 10de:",
               "systemctl is-active nvidia-fabricmanager ; dpkg-query nvidia-fabricmanager*", "dcgmi discovery -l"],
    "rdma": ["/sys/class/infiniband/*/ports/*", "nvidia-smi topo -m (rail map)",
             "taskset -c <NIC-local CPUs> ib_write_bw -d <rail k> ... 127.0.0.1 <- rail k+1 (ring, 5 s each)",
             "ib_write_bw --use_cuda=<rail GPU> --use_cuda_dmabuf, else the peermem path (if perftest has CUDA)"],
    "health": ["before and after the load: nvidia-smi ECC + remapped rows, dmesg XIDs, IB counters, "
               "nvidia-smi nvlink -e"],
    "dcgm": ["dcgmi diag -r <level> -j (starts nv-hostengine only if none is running)"],
    "burnin": ["dcgmi dmon -e 152,158,159,151 -c 1 (GPU temperature limits)",
               "dcgmi diag -r targeted_power -p targeted_power.test_duration=<burnin.minutes*60> -j",
               "dcgmi dmon -e 150,140,155,100 -d 5000 alongside (temperatures, power, SM clock)",
               "nvidia-smi --query-gpu=index,clocks_event_reasons_counters.* before and after (throttling)",
               "skipped when the manifest has no [burnin] table"],
    "nccl": ["all_reduce_perf -b 8 -e 8G -f 2 -g <gpus> -n 20 -w 5"],
    "fio": ["fio seq write + read in [storage].path (only if the manifest has [storage])"],
}


def print_plan(cls, spec, source, stages, args):
    perf = spec.get("perf", {})
    print(f"class {cls} from {source}")
    print(f"  expect {spec['gpus']}x {spec['gpu_name']}, driver {spec['driver']}, CUDA {spec.get('cuda', '-')}, "
          f"NCCL {spec.get('nccl', '-')}, {spec['nvlinks_total']} NVLinks, {spec['nvswitches']} NVSwitches, "
          f"{spec['rdma_nics']}x {spec.get('rdma_link_layer', 'InfiniBand')} @ {spec['rdma_rate_gbps']} Gb/s")
    print("  floors " + (", ".join(f"{k}={v}" for k, v in perf.items()) or "none"))
    if spec.get("gpu_power_limit_w") or spec.get("thermal") or spec.get("burnin"):
        print(f"  power limit {spec.get('gpu_power_limit_w', '-')} W, thermal "
              + (", ".join(f"{k}={v}" for k, v in (spec.get("thermal") or {}).items()) or "-")
              + ", burn-in " + (", ".join(f"{k}={v}" for k, v in (spec.get("burnin") or {}).items()) or "none"))
    print(f"tools dir {args.tools_dir} (install {'off' if args.no_install else 'on'}), "
          f"evidence {args.out or DEFAULT_OUT + '/<run_id>'}, DCGM level {args.dcgm_level}")
    print("prereq: root ; nvidia-smi ; nvidia-smi --query-compute-apps (GPUs must be idle unless --force)")
    for s in stages:
        for line in PLAN[s]:
            print(f"{s}: {line}")
