"""RDMA NIC ports, GPU-to-NIC rail map, and per-rail RDMA write bandwidth through the switch."""

from . import parse

BW_ARGS = ["-F", "--report_gbits", "-D", "5", "-s", "65536"]
BASE_PORT = 18515


def run_ports(ctx):
    ctx.stage("rdma", "ports and rail map")
    spec, f = ctx.spec, ctx.facts
    rails = f["rails"]
    layer = spec.get("rdma_link_layer", "InfiniBand")
    ctx.add("rdma.nic_count", "pass" if len(rails) == spec["rdma_nics"] else "fail", None, len(rails),
            spec["rdma_nics"], f"devices with link layer {layer}")
    bad = 0
    for dev in rails:
        for port, p in sorted(f["ib"][dev]["ports"].items()):
            problems = []
            if "ACTIVE" not in p["state"].upper():
                problems.append(f"state {p['state']}")
            if layer == "InfiniBand" and "LINKUP" not in p["phys_state"].upper().replace(" ", ""):
                problems.append(f"phys {p['phys_state']}")
            if p["rate"] != spec["rdma_rate_gbps"]:
                problems.append(f"rate {p['rate_text'] or p['rate']}")
            if problems:
                bad += 1
                ctx.add("rdma.port", "fail", f"{dev}/{port}", "; ".join(problems),
                        f"ACTIVE LinkUp {spec['rdma_rate_gbps']} Gb/s")
    if rails and not bad:
        ctx.add("rdma.port", "pass", None, f"{len(rails)} ports ACTIVE at {spec['rdma_rate_gbps']} Gb/s",
                f"ACTIVE LinkUp {spec['rdma_rate_gbps']} Gb/s")
    rail_map(ctx)


def rail_map(ctx):
    tp = ctx.facts.get("topo")
    if not tp:
        ctx.add("rdma.rail_map", "error", None, None, "one NIC per GPU", "nvidia-smi topo -m failed")
        return
    rmap = parse.rail_map(tp, set(ctx.facts["rails"]))
    ctx.facts["rail_map"] = rmap
    if len(rmap) != ctx.spec["gpus"]:
        ctx.add("rdma.rail_map", "error", None, len(rmap), ctx.spec["gpus"],
                "topology matrix did not list every GPU", evidence="raw/nvidia-smi-topo.out")
        return
    seen, bad = {}, 0
    for gpu, dev in sorted(rmap.items()):
        if dev is None:
            bad += 1
            ctx.add("rdma.rail_map", "fail", f"GPU{gpu}", None, "PIX/PXB NIC",
                    "no RDMA NIC behind the same PCIe switch", evidence="raw/nvidia-smi-topo.out")
        elif dev in seen:
            bad += 1
            ctx.add("rdma.rail_map", "fail", f"GPU{gpu}", dev, "own NIC", f"shares {dev} with GPU{seen[dev]}",
                    evidence="raw/nvidia-smi-topo.out")
        else:
            seen[dev] = gpu
    if rmap and not bad:
        ctx.add("rdma.rail_map", "pass", None, {f"GPU{g}": d for g, d in sorted(rmap.items())},
                "one NIC per GPU", evidence="raw/nvidia-smi-topo.out")


def run_bandwidth(ctx):
    rails = [d for d in ctx.facts["rails"]
             if any("ACTIVE" in p["state"].upper() for p in ctx.facts["ib"][d]["ports"].values())]
    if ctx.spec.get("rdma_link_layer", "InfiniBand") != "InfiniBand":
        ctx.add("rdma.write_bw", "skip", None, None, None, "RoCE loopback needs per-rail IPs and GID setup (v1: IB only)")
        if gdr_required(ctx):
            ctx.add("rdma.gdr_write_bw", "error", None, None, None,
                    "host.gpudirect requires a GPUDirect RDMA test, which v1 runs on InfiniBand only")
        return
    if len(rails) < 2:
        ctx.add("rdma.write_bw", "error", None, len(rails), ">= 2 active ports", "nothing to pair for loopback")
        return
    if not ctx.which("ib_write_bw"):
        ctx.add("rdma.write_bw", "error", None, None, None, "ib_write_bw not found")
        return
    ctx.stage("rdma", f"ib_write_bw loopback ring over {len(rails)} rails, 5 s each")
    floor = ctx.perf("rdma_write_bw_gbps")
    for k, server in enumerate(rails):
        client = rails[(k + 1) % len(rails)]
        bw, why, ev = _pair(ctx, f"ib_write_bw-{server}-{client}", server, client, BASE_PORT + k, [], [])
        _verdict(ctx, "rdma.write_bw", f"{server}<-{client}", bw, floor, why, ev)
    gpudirect(ctx, rails, floor)


# GPUDirect paths, tried in this order: DMA-BUF (open kernel module, no extra module) then nvidia-peermem.
GDR_PATHS = (("dmabuf", ["--use_cuda_dmabuf"]), ("peermem", []))


def gdr_required(ctx):
    return bool(ctx.table("host").get("gpudirect"))


def gpudirect(ctx, rails, floor):
    """GPUDirect RDMA: each side's buffers live on its rail-local GPU. A pair that cannot register GPU memory is
    an error (GPUDirect does not work); bandwidth below the host-memory floor is a warning until measured.
    With host.gpudirect = true, a test that cannot run (perftest without CUDA) is an error, not a skip."""
    helptext = ctx.run("ib_write_bw-help", ["ib_write_bw", "--help"], timeout=30)
    text = helptext.out + helptext.err
    if "use_cuda" not in text:
        ctx.add("rdma.gdr_write_bw", "error" if gdr_required(ctx) else "skip", None, None, None,
                "perftest built without CUDA support (Ubuntu's perftest has none; DOCA/MLNX_OFED perftest has)",
                helptext.evidence)
        return
    paths = [p for p in GDR_PATHS if p[0] != "dmabuf" or "use_cuda_dmabuf" in text]
    by_dev = {d: g for g, d in (ctx.facts.get("rail_map") or {}).items() if d}
    chosen = None                      # the first path that works is used for the remaining pairs
    for k, server in enumerate(rails):
        client = rails[(k + 1) % len(rails)]
        target = f"{server}<-{client}"
        if server not in by_dev or client not in by_dev:
            ctx.add("rdma.gdr_write_bw", "skip", target, None, None, "no rail-local GPU")
            continue
        tried = []
        for path, flags in ([chosen] + [p for p in paths if p != chosen] if chosen else paths):
            bw, why, ev = _pair(ctx, f"ib_write_bw-gdr-{path}-{server}-{client}", server, client,
                                BASE_PORT + 100 + k, [f"--use_cuda={by_dev[server]}", *flags],
                                [f"--use_cuda={by_dev[client]}", *flags])
            if bw is not None:
                chosen = (path, flags)
                break
            tried.append(f"{path}: {why}")
        if bw is None:
            ctx.add("rdma.gdr_write_bw", "error", target, None, floor,
                    "GPUDirect failed on every path (" + "; ".join(tried) + ")", ev)
            continue
        note = f"via {chosen[0]}" + (f" ({'; '.join(tried)})" if tried else "")
        _verdict(ctx, "rdma.gdr_write_bw", target, bw, floor, None, ev, below="warn", note=note)


def _pinned(ctx, dev, cmd):
    """Run cmd on the CPUs local to the NIC, so its buffers are allocated on the NIC's NUMA node."""
    cpus = ctx.sys.read(f"/sys/class/infiniband/{dev}/device/local_cpulist")
    if cpus and ctx.which("taskset"):
        return ["taskset", "-c", cpus, *cmd], True
    return cmd, False


def _pair(ctx, name, server, client, port, s_extra, c_extra):
    scmd, s_pin = _pinned(ctx, server, ["ib_write_bw", "-d", server, *BW_ARGS, "-p", str(port), *s_extra])
    srv = ctx.spawn(f"{name}-server", scmd)
    if srv is None:
        return None, "ib_write_bw could not be started", None
    try:
        ctx.sleep()
        ccmd, c_pin = _pinned(ctx, client, ["ib_write_bw", "-d", client, *BW_ARGS, "-p", str(port), *c_extra,
                                            "127.0.0.1"])
        cli = ctx.run(name, ccmd, timeout=90)
    finally:
        srv.finish()
    ctx.facts["rdma_pinned"] = s_pin and c_pin          # read by _verdict for this pair
    bw = parse.perftest_bw(cli.out)
    return bw, (None if bw is not None else cli.why()), cli.evidence


def _verdict(ctx, check, target, bw, floor, why, ev, below="fail", note=""):
    pin = "NUMA-pinned" if ctx.facts.get("rdma_pinned") else "not pinned"
    extra = f"; {note}" if note else ""
    if bw is None:
        ctx.add(check, "error", target, None, floor, why or "no bandwidth row in output", ev)
    elif floor is None:
        ctx.add(check, "info", target, bw, None, f"Gb/s; no floor in manifest; {pin}{extra}", ev)
    else:
        ctx.add(check, "pass" if bw >= floor else below, target, bw, floor,
                f"Gb/s, 64 KiB, single QP, {pin}{extra}", ev)
