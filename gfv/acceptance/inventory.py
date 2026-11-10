"""Manifest diff: compare what the node has with what its class manifest says it should have."""

from . import parse
from .parse import BDF

LATER_TIER = "needs BMC/Redfish (later tier)"


def run(ctx):
    ctx.stage("inventory", "manifest diff")
    spec, f = ctx.spec, ctx.facts
    gpus = f["gpus"]

    _eq(ctx, "inventory.gpu_count", len(gpus), spec["gpus"], evidence="raw/nvidia-smi-query-gpu.out")
    names = sorted({g["name"] for g in gpus})
    bad = [g for g in gpus if g["name"] != spec["gpu_name"]]
    for g in bad:
        ctx.add("inventory.gpu_name", "fail", f"GPU{g['index']}", g["name"], spec["gpu_name"])
    if gpus and not bad:
        ctx.add("inventory.gpu_name", "pass", None, names[0], spec["gpu_name"])
    _eq(ctx, "inventory.driver", f.get("driver"), spec["driver"], evidence="raw/nvidia-smi.out")
    if "cuda" in spec:
        _eq(ctx, "inventory.cuda", f.get("cuda"), spec["cuda"], evidence="raw/nvidia-smi.out")

    mig = [g for g in gpus if g["mig.mode.current"].lower() == "enabled"]
    for g in mig:
        ctx.add("inventory.mig", "fail", f"GPU{g['index']}", "Enabled", "Disabled", "MIG splits the GPU")
    if gpus and not mig:
        ctx.add("inventory.mig", "pass", None, "Disabled", "Disabled")
    no_ecc = [g for g in gpus if g["ecc.mode.current"].lower() != "enabled"]
    for g in no_ecc:
        ctx.add("inventory.ecc_mode", "fail", f"GPU{g['index']}", g["ecc.mode.current"], "Enabled",
                "ECC off: memory errors go unreported")
    if gpus and not no_ecc:
        ctx.add("inventory.ecc_mode", "pass", None, "Enabled", "Enabled")
    off = [g for g in gpus if g["persistence_mode"].lower() == "disabled"]
    for g in off:
        ctx.add("inventory.persistence_mode", "warn", f"GPU{g['index']}", "Disabled", "Enabled")
    if gpus and not off:
        ctx.add("inventory.persistence_mode", "pass", None, "Enabled", "Enabled")

    pcie_widths(ctx)
    numa = len(ctx.sys.glob("/sys/devices/system/node/node[0-9]*"))
    _eq(ctx, "inventory.numa_nodes", numa, spec["numa_nodes"])
    host(ctx)
    firmware(ctx)
    ofed = ctx.run("ofed_info", ["ofed_info", "-s"], timeout=30)
    if ofed.ok:
        ctx.rec.versions["ofed"] = parse.ofed_version(ofed.out)
        ctx.add("inventory.ofed", "info", None, ctx.rec.versions["ofed"], None, evidence="raw/ofed_info.out")
    else:
        ctx.add("inventory.ofed", "info", None, None, None, f"ofed_info unavailable ({ofed.why()})")
    ctx.rec.versions["bios"] = ctx.sys.read("/sys/class/dmi/id/bios_version")


def pcie_widths(ctx):
    want = ctx.spec["pcie_width"]
    bad = 0
    for g in ctx.facts["gpus"]:
        cur, mx = parse.as_int(g["pcie.link.width.current"]), parse.as_int(g["pcie.link.width.max"])
        if cur is None or cur < (mx or 0) or cur != want:
            bad += 1
            ctx.add("inventory.pcie_width", "fail", f"GPU{g['index']}", cur, want,
                    f"max x{mx}; GPUs keep width at idle, so a narrow link is a training or seating fault")
    for dev in ctx.facts["rails"]:
        d = ctx.facts["ib"][dev]
        if d["width"] is None or d["width"] < (d["max_width"] or 0) or d["width"] != want:
            bad += 1
            ctx.add("inventory.pcie_width", "fail", dev, d["width"], want, f"max x{d['max_width']}")
    if not bad and (ctx.facts["gpus"] or ctx.facts["rails"]):
        ctx.add("inventory.pcie_width", "pass", None, want, want,
                f"{len(ctx.facts['gpus'])} GPUs + {len(ctx.facts['rails'])} RDMA NICs at x{want}")


def host(ctx):
    h = ctx.table("host")
    cmdline = ctx.sys.read("/proc/cmdline", "") or ""
    for needle in h.get("kernel_cmdline_contains", []):
        ok = needle in cmdline
        ctx.add("inventory.kernel_cmdline", "pass" if ok else "fail", needle, ok, True,
                "" if ok else "not in /proc/cmdline")
    if "secure_boot" in h:
        sb = ctx.run("mokutil", ["mokutil", "--sb-state"], timeout=30)
        state = parse.mokutil(sb.out + sb.err)
        if state is None:
            ctx.add("inventory.secure_boot", "error", None, None, h["secure_boot"],
                    f"cannot read Secure Boot state ({(sb.out + sb.err).strip()[:120] or sb.why()})",
                    evidence="raw/mokutil.out")
        else:
            ctx.add("inventory.secure_boot", "pass" if state == h["secure_boot"] else "fail", None,
                    state, h["secure_boot"], evidence="raw/mokutil.out")
    if "pcie_acs" in h:
        acs(ctx, h["pcie_acs"])
    if "nvidia_peermem" in h:
        loaded = any(line.split()[:1] == ["nvidia_peermem"]
                     for line in (ctx.sys.read("/proc/modules", "") or "").splitlines())
        ctx.add("inventory.nvidia_peermem", "pass" if loaded == h["nvidia_peermem"] else "fail", None,
                loaded, h["nvidia_peermem"], "" if loaded else "GPUDirect RDMA module not loaded")
    if "power_profile" in h:
        ctx.add("inventory.power_profile", "skip", None, None, h["power_profile"], LATER_TIER)


def switch_ports(sysfs, devpath, bdf=None):
    """PCIe switch ports between the root port and a device, from its sysfs path; None if unresolvable.
    The root port (first BDF under the host bridge) is excluded: same-switch P2P never reaches it."""
    if not sysfs.exists(devpath):
        return None
    real = sysfs.realpath(devpath)
    parts = [p.lower() for p in real.split("/") if BDF.match(p)]
    if not parts or (bdf and parts[-1] != bdf):
        return None
    return parts[1:-1]


def acs(ctx, want):
    out = ctx.run("lspci-vvv", ["lspci", "-D", "-vvv"], timeout=60)
    if not out.ok:
        ctx.add("inventory.pcie_acs", "error", None, None, want, out.why())
        return
    flags, denied = parse.lspci_acs(out.out)
    devices = [(f"GPU{g['index']}", f"/sys/bus/pci/devices/{g['bus']}", g["bus"]) for g in ctx.facts["gpus"]]
    devices += [(dev, f"/sys/class/infiniband/{dev}/device", None) for dev in ctx.facts["rails"]]
    bad, checked, hidden, unresolved = [], set(), set(), []
    for name, path, bdf in devices:
        ports = switch_ports(ctx.sys, path, bdf)
        if ports is None:
            unresolved.append(name)
            continue
        for br in ports:
            if br in denied:
                hidden.add(br)
            if br not in flags:
                continue
            checked.add(br)
            redirect = flags[br].get("ReqRedir") or flags[br].get("CmpltRedir")
            if (want == "disabled") == bool(redirect):
                bad.append((name, br, redirect))
    ev = "raw/lspci-vvv.out"
    if hidden:
        ctx.add("inventory.pcie_acs", "error", None, None, want, "lspci hides capabilities (not root?)", ev)
        return
    if unresolved:
        ctx.add("inventory.pcie_acs", "error", None, unresolved, want, "no sysfs PCI path for these devices", ev)
    for name, br, redirect in bad:
        ctx.add("inventory.pcie_acs", "fail", f"{name} via {br}", "enabled" if redirect else "disabled", want,
                "ReqRedir/CmpltRedir set: P2P and GPUDirect traffic detours through the root complex", ev)
    if not checked:
        ctx.add("inventory.pcie_acs", "error", None, None, want,
                "no ACS control found on any PCIe switch port above the GPUs and NICs", ev)
    elif not bad and not unresolved:
        ctx.add("inventory.pcie_acs", "pass", None, want, want,
                f"{len(checked)} switch ports above {len(devices)} GPUs/NICs checked", ev)


def firmware(ctx):
    fw = ctx.table("firmware")
    found = {
        "bios": {"host": ctx.sys.read("/sys/class/dmi/id/bios_version")},
        "gpu_vbios": {f"GPU{g['index']}": g["vbios_version"] for g in ctx.facts["gpus"]},
        "nic": {d: ctx.facts["ib"][d]["fw_ver"] for d in ctx.facts["rails"]},
    }
    for kind, values in found.items():
        allow = fw.get(kind, [])
        distinct = sorted({v for v in values.values() if v})
        if not allow:
            ctx.add(f"inventory.firmware.{kind}", "info", None, distinct, None, "no allowlist; recorded only")
            continue
        bad = {t: v for t, v in values.items() if v not in allow}
        for t, v in bad.items():
            ctx.add(f"inventory.firmware.{kind}", "fail", t, v, allow)
        if not bad:
            ctx.add(f"inventory.firmware.{kind}", "pass", None, distinct, allow)
    if "bmc" in fw:
        ctx.add("inventory.firmware.bmc", "skip", None, None, fw["bmc"] or None, LATER_TIER)


def _eq(ctx, check, value, expected, detail="", evidence=None):
    ctx.add(check, "pass" if value == expected else "fail", None, value, expected, detail, evidence)
