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
        bw, why = _pair(ctx, f"ib_write_bw-{server}-{client}", server, client, BASE_PORT + k, [], [])
        _verdict(ctx, "rdma.write_bw", f"{server}<-{client}", bw, floor, why)
    gpudirect(ctx, rails, floor)


def gpudirect(ctx, rails, floor):
    """GPUDirect RDMA variant: each side's buffers live on its rail-local GPU (advisory in v1)."""
    helptext = ctx.run("ib_write_bw-help", ["ib_write_bw", "--help"], timeout=30)
    if "use_cuda" not in helptext.out + helptext.err:
        ctx.add("rdma.gdr_write_bw", "skip", None, None, None, "perftest built without CUDA support")
        return
    by_dev = {d: g for g, d in (ctx.facts.get("rail_map") or {}).items() if d}
    for k, server in enumerate(rails):
        client = rails[(k + 1) % len(rails)]
        if server not in by_dev or client not in by_dev:
            ctx.add("rdma.gdr_write_bw", "skip", f"{server}<-{client}", None, None, "no rail-local GPU")
            continue
        bw, why = _pair(ctx, f"ib_write_bw-gdr-{server}-{client}", server, client, BASE_PORT + 100 + k,
                        [f"--use_cuda={by_dev[server]}"], [f"--use_cuda={by_dev[client]}"])
        _verdict(ctx, "rdma.gdr_write_bw", f"{server}<-{client}", bw, floor, why, below="warn")


def _pair(ctx, name, server, client, port, s_extra, c_extra):
    srv = ctx.spawn(f"{name}-server", ["ib_write_bw", "-d", server, *BW_ARGS, "-p", str(port), *s_extra])
    if srv is None:
        return None, "ib_write_bw not found"
    ctx.sleep()
    cli = ctx.run(name, ["ib_write_bw", "-d", client, *BW_ARGS, "-p", str(port), *c_extra, "127.0.0.1"],
                  timeout=90)
    srv.finish()
    bw = parse.perftest_bw(cli.out)
    return bw, (None if bw is not None else cli.why())


def _verdict(ctx, check, target, bw, floor, why, below="fail"):
    ev = None
    if bw is None:
        ctx.add(check, "error", target, None, floor, why or "no bandwidth row in output")
    elif floor is None:
        ctx.add(check, "info", target, bw, None, "Gb/s; no floor in manifest", ev)
    else:
        ctx.add(check, "pass" if bw >= floor else below, target, bw, floor, "Gb/s, 64 KiB, single QP", ev)
