"""Shared state for one acceptance run, and node facts discovered once and read by every stage."""

import os
import re
import shutil
import sys
import time

from ..common.results import utcnow
from . import parse


class Ctx:
    def __init__(self, args, cls, spec, out_dir, proc, sysfs, rec, settle=1.5):
        self.args, self.cls, self.spec = args, cls, spec
        self.out_dir, self.proc, self.sys, self.rec = out_dir, proc, sysfs, rec
        self.settle = settle
        self.facts = {}
        self.tool_path = []        # extra PATH entries (installed tools)
        self.ld_path = []          # extra LD_LIBRARY_PATH entries (NCCL)
        self.nccl_tests = None     # directory with build/all_reduce_perf
        self.notes = []

    # --- spec helpers -----------------------------------------------------------------------
    def table(self, name):
        return self.spec.get(name, {}) or {}

    def perf(self, key):
        return self.table("perf").get(key)

    # --- tools ------------------------------------------------------------------------------
    def env_extra(self):
        env = {}
        if self.tool_path:
            env["PATH"] = os.pathsep.join(self.tool_path + [self.proc.env.get("PATH", "")])
        if self.ld_path:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(self.ld_path + [self.proc.env.get("LD_LIBRARY_PATH", "")])
        return env

    def which(self, tool):
        path = os.pathsep.join(self.tool_path + [self.proc.env.get("PATH", "")])
        return shutil.which(tool, path=path)

    def run(self, name, cmd, timeout=120):
        return self.proc.run(name, cmd, timeout=timeout, env_extra=self.env_extra())

    def spawn(self, name, cmd):
        return self.proc.spawn(name, cmd, env_extra=self.env_extra())

    # --- output -----------------------------------------------------------------------------
    def stage(self, name, note=""):
        print(f"[{utcnow()}] stage {name}" + (f": {note}" if note else ""), file=sys.stderr, flush=True)

    def add(self, *a, **kw):
        return self.rec.add(*a, **kw)

    def sleep(self, seconds=None):
        time.sleep(self.settle if seconds is None else seconds)


def ib_devices(sysfs):
    """RDMA devices from /sys/class/infiniband with port state, rate, counters and PCIe link width."""
    devs = {}
    for dev in sysfs.listdir("/sys/class/infiniband"):
        base = f"/sys/class/infiniband/{dev}"
        ports = {}
        for port in sysfs.listdir(f"{base}/ports"):
            pb = f"{base}/ports/{port}"
            rate = sysfs.read(f"{pb}/rate", "") or ""
            counters = {}
            for c in sysfs.listdir(f"{pb}/counters"):
                v = parse.as_int(sysfs.read(f"{pb}/counters/{c}"))
                if v is not None:
                    counters[c] = v
            ports[port] = {
                "state": sysfs.read(f"{pb}/state", ""), "phys_state": sysfs.read(f"{pb}/phys_state", ""),
                "rate": parse.as_int(rate.split()[0]) if rate.split() else None, "rate_text": rate,
                "link_layer": sysfs.read(f"{pb}/link_layer", ""), "counters": counters,
            }
        devs[dev] = {
            "fw_ver": sysfs.read(f"{base}/fw_ver"), "ports": ports,
            "width": parse.as_int(sysfs.read(f"{base}/device/current_link_width")),
            "max_width": parse.as_int(sysfs.read(f"{base}/device/max_link_width")),
        }
    return devs


def discover(ctx):
    """Cheap reads every stage needs. Runs even when the inventory stage is skipped."""
    f = ctx.facts
    hdr = ctx.run("nvidia-smi", ["nvidia-smi"], timeout=60)
    f["smi_ok"] = hdr.ok
    f["smi_why"] = hdr.why()
    f.update(parse.smi_header(hdr.out))
    f["gpus"] = query_gpus(ctx, "nvidia-smi-query-gpu")
    tp = ctx.run("nvidia-smi-topo", ["nvidia-smi", "topo", "-m"], timeout=60)
    f["topo"] = parse.topo(tp.out) if tp.ok else None
    f["ib"] = ib_devices(ctx.sys)
    layer = ctx.spec.get("rdma_link_layer", "InfiniBand")
    f["rails"] = sorted((d for d, v in f["ib"].items()
                         if any(p["link_layer"] == layer for p in v["ports"].values())),
                        key=_natural)
    v = ctx.rec.versions
    v["driver"] = f.get("driver")
    v["cuda"] = f.get("cuda")
    v["kernel"] = ctx.sys.read("/proc/sys/kernel/osrelease")
    v["os"] = ctx.sys.os_release().get("PRETTY_NAME")


def query_gpus(ctx, name):
    q = ctx.run(name, ["nvidia-smi", "--query-gpu=" + ",".join(parse.GPU_FIELDS),
                       "--format=csv,noheader,nounits"], timeout=60)
    if not q.ok:
        return []
    try:
        return parse.gpu_csv(q.out)
    except ValueError:
        return []


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
