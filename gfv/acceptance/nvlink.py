"""NVLink links, NVSwitch fabric state, NVSwitch count and Fabric Manager."""

from . import parse


def run(ctx):
    spec = ctx.spec
    if spec["nvlinks_total"] == 0 and spec["nvswitches"] == 0:
        ctx.add("nvlink", "skip", None, None, None, "class has no NVLink")
        return
    ctx.stage("nvlink")
    st = ctx.run("nvidia-smi-nvlink-status", ["nvidia-smi", "nvlink", "-s"], timeout=60)
    if not st.ok:
        ctx.add("nvlink.links_active", "error", None, None, spec["nvlinks_total"], st.why())
    else:
        links = parse.nvlink_status(st.out)
        active = sum(v["active"] for v in links.values())
        for gpu, v in sorted(links.items()):
            if v["inactive"]:
                ctx.add("nvlink.links_active", "fail", f"GPU{gpu}", v["active"], v["active"] + v["inactive"],
                        f"{v['inactive']} inactive", evidence=st.evidence)
        ctx.add("nvlink.links_total", "pass" if active == spec["nvlinks_total"] else "fail", None, active,
                spec["nvlinks_total"], evidence=st.evidence)

    if spec["nvswitches"] > 0:
        fabric_state(ctx)
        fabric_manager(ctx)
    dev = spec.get("nvswitch_pci_device")
    if dev:
        ls = ctx.run("lspci-nvidia", ["lspci", "-nn", "-d", "10de:"], timeout=60)
        if ls.ok:
            n = parse.lspci_count(ls.out, dev)
            ctx.add("nvlink.nvswitch_count", "pass" if n == spec["nvswitches"] else "fail", None, n,
                    spec["nvswitches"], f"PCI device 10de:{dev}", evidence=ls.evidence)
        else:
            ctx.add("nvlink.nvswitch_count", "error", None, None, spec["nvswitches"], ls.why())
    elif spec["nvswitches"] > 0:
        ctx.add("nvlink.nvswitch_count", "skip", None, None, spec["nvswitches"],
                "set nvswitch_pci_device in the manifest to count NVSwitches")
    if spec["nvswitches"] > 0 and ctx.which("dcgmi"):
        disc = ctx.run("dcgmi-discovery", ["dcgmi", "discovery", "-l"], timeout=60)
        seen = parse.dcgmi_nvswitches(disc.out) if disc.ok else None
        if seen == 0:
            ctx.add("nvlink.dcgm_nvswitch_visible", "warn", None, 0, spec["nvswitches"],
                    "DCGM sees no NVSwitch (libnvidia-nscq missing?); DCGM NVSwitch health is blind",
                    evidence=disc.evidence)
        elif seen is not None:
            ctx.add("nvlink.dcgm_nvswitch_visible", "pass", None, seen, spec["nvswitches"], evidence=disc.evidence)


def fabric_state(ctx):
    q = ctx.run("nvidia-smi-q", ["nvidia-smi", "-q"], timeout=120)
    if not q.ok:
        ctx.add("nvlink.fabric_state", "error", None, None, "Completed/Success", q.why())
        return
    fab = parse.fabric(q.out)
    bad = 0
    for g in ctx.facts["gpus"]:
        s = fab.get(g["bus"], {})
        val = f"{s.get('state')}/{s.get('status')}"
        if val != "Completed/Success":
            bad += 1
            ctx.add("nvlink.fabric_state", "fail", f"GPU{g['index']}", val, "Completed/Success",
                    "NVSwitch fabric not trained (Fabric Manager down or at the wrong version?)",
                    evidence=q.evidence)
    if not bad and ctx.facts["gpus"]:
        ctx.add("nvlink.fabric_state", "pass", None, "Completed/Success", "Completed/Success", evidence=q.evidence)


def fabric_manager(ctx):
    act = ctx.run("systemctl-fabricmanager", ["systemctl", "is-active", "nvidia-fabricmanager"], timeout=30)
    state = act.out.strip() or act.why()
    ctx.add("nvlink.fabricmanager_active", "pass" if state == "active" else "fail", None, state, "active",
            evidence=act.evidence)
    pkg = ctx.run("dpkg-fabricmanager", ["dpkg-query", "-W", "-f", "${Package} ${Version}\\n",
                                         "nvidia-fabricmanager*"], timeout=30)
    versions = [line.split()[1] for line in pkg.out.splitlines() if len(line.split()) == 2]
    if not versions:
        ctx.add("nvlink.fabricmanager_version", "error", None, None, ctx.facts.get("driver"),
                f"no nvidia-fabricmanager package found ({pkg.why()})")
        return
    fm = versions[0].split("-")[0]
    ctx.rec.versions["fabricmanager"] = fm
    ctx.add("nvlink.fabricmanager_version", "pass" if fm == ctx.facts.get("driver") else "fail", None, fm,
            ctx.facts.get("driver"), "Fabric Manager must match the driver exactly", evidence=pkg.evidence)
