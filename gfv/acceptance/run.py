"""L2 acceptance runner: stage order, fail-closed verdict, evidence directory and exit codes."""

import json
import os
import secrets
import socket
import sys
import traceback

from .. import build_info
from ..common import install, manifest
from ..common.proc import Proc
from ..common.results import Recorder, utcnow
from ..common.sysinfo import SysFS
from . import dcgm, fio, health, inventory, nccl, nvlink, parse, rdma
from .context import Ctx, discover

STAGES = ("install", "inventory", "nvlink", "rdma", "health", "dcgm", "nccl", "fio")
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
    p.add_argument("--node", default=socket.gethostname(), help="node name in results (default: hostname)")
    p.add_argument("--cluster", help="cluster label for results")
    p.add_argument("--rack", help="rack label for results")
    p.add_argument("--leaf", help="leaf-switch label for results")
    p.add_argument("--force", action="store_true", help="run even if GPU processes are running")
    p.add_argument("--fail-fast", action="store_true", help="skip the heavy stages once a cheap check failed")
    p.add_argument("--dry-run", action="store_true", help="validate the manifest and print the plan; run nothing")


def stages_from(args):
    def parse_list(v):
        names = [s.strip() for s in (v or "").split(",") if s.strip()]
        bad = [n for n in names if n not in STAGES]
        if bad:
            raise ValueError(f"unknown stage(s) {bad}; known: {', '.join(STAGES)}")
        return names
    only, skip = parse_list(args.only), parse_list(args.skip)
    return [s for s in STAGES if (not only or s in only) and s not in skip]


def accept(args):
    try:
        stages = stages_from(args)
        cls, spec, source, _ = manifest.load(args.manifest, args.cls)
    except (ValueError, manifest.ManifestError) as e:
        print(f"gfv: {e}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    if args.no_install:
        stages = [s for s in stages if s != "install"]
    if args.dry_run:
        print_plan(cls, spec, source, stages, args)
        return EXIT_PASS
    if os.geteuid() != 0 and not os.environ.get("GFV_ALLOW_NONROOT"):
        print("gfv: run as root (DCGM diagnostics, kernel log, PCI config space, tool install)", file=sys.stderr)
        return EXIT_CANNOT_RUN

    run_id = f"l2-{utcnow().replace('-', '').replace(':', '')}-{secrets.token_hex(2)}"
    out_dir = args.out or os.path.join(DEFAULT_OUT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"class": cls, "source": source, "spec": spec}, f, indent=2)
    context = {"run_id": run_id, "cluster": args.cluster, "node": args.node, "class": cls, "rack": args.rack,
               "leaf": args.leaf, "pod": None, "tier": "L2"}
    rec = Recorder(context, out_dir)
    info = build_info()
    rec.versions["gfv"] = f"{info['version']} ({info['commit']})"
    ctx = Ctx(args, cls, spec, out_dir, Proc(out_dir), SysFS(os.environ.get("GFV_SYSROOT", "/")), rec,
              settle=float(os.environ.get("GFV_SETTLE_SECONDS", "1.5")))
    print(f"gfv L2 acceptance {run_id}: node {args.node}, class {cls} ({source}), stages {','.join(stages)}",
          file=sys.stderr)
    code = EXIT_FAIL
    try:
        code = run_stages(ctx, stages)
    except KeyboardInterrupt:
        rec.add("run", "error", None, None, None, "interrupted")
        code = EXIT_FAIL
    finally:
        rec.finish({"notes": ctx.notes, "finished": utcnow()})
        verdict = "PASS" if code == EXIT_PASS else ("CANNOT RUN" if code == EXIT_CANNOT_RUN else "FAIL")
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
        existing = nccl.find_installed(args.tools_dir)
        if existing:
            install.load_existing(ctx, existing)

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
    "inventory": ["nvidia-smi --query-gpu=...", "/proc/cmdline, /sys/devices/system/node", "mokutil --sb-state",
                  "lspci -D -vvv (ACS on bridges above GPUs)", "/proc/modules (nvidia_peermem)", "ofed_info -s",
                  "/sys/class/dmi/id/bios_version, /sys/class/infiniband/*/fw_ver"],
    "nvlink": ["nvidia-smi nvlink -s", "nvidia-smi -q (Fabric state)", "lspci -nn -d 10de:",
               "systemctl is-active nvidia-fabricmanager ; dpkg-query nvidia-fabricmanager*", "dcgmi discovery -l"],
    "rdma": ["/sys/class/infiniband/*/ports/*", "nvidia-smi topo -m (rail map)",
             "ib_write_bw -d <rail k> ... 127.0.0.1 <- rail k+1 (ring, 5 s each)",
             "ib_write_bw --use_cuda=<rail GPU> (if perftest has CUDA)"],
    "health": ["before and after the load: nvidia-smi ECC + remapped rows, dmesg XIDs, IB counters, "
               "nvidia-smi nvlink -e"],
    "dcgm": ["dcgmi diag -r <level> -j (starts nv-hostengine only if none is running)"],
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
    print(f"tools dir {args.tools_dir} (install {'off' if args.no_install else 'on'}), "
          f"evidence {args.out or DEFAULT_OUT + '/<run_id>'}, DCGM level {args.dcgm_level}")
    print("prereq: root ; nvidia-smi ; nvidia-smi --query-compute-apps (GPUs must be idle unless --force)")
    for s in stages:
        for line in PLAN[s]:
            print(f"{s}: {line}")
