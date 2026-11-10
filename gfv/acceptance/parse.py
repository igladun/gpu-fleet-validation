"""Parsers for tool output. Pure functions; every one is tested against recorded output in tests/."""

import csv
import io
import json
import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BDF = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")

GPU_FIELDS = [
    "index", "uuid", "name", "pci.bus_id", "driver_version", "vbios_version", "serial",
    "persistence_mode", "mig.mode.current", "pcie.link.width.current", "pcie.link.width.max",
    "pcie.link.gen.current", "pcie.link.gen.max", "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.total", "compute_cap",
]
REMAP_FIELDS = ["gpu_bus_id", "remapped_rows.correctable", "remapped_rows.uncorrectable",
                "remapped_rows.pending", "remapped_rows.failure"]
APP_FIELDS = ["pid", "process_name", "gpu_bus_id", "used_memory"]


def strip_ansi(text):
    return ANSI.sub("", text or "")


def norm_bus(bus):
    """'00000000:0A:00.0' or '0000:0a:00.0' -> '0000:0a:00.0' (the sysfs spelling)."""
    bus = bus.strip().lower()
    if bus.startswith("pci:"):
        bus = bus[4:]
    parts = bus.split(":")
    if len(parts) == 3:
        parts[0] = parts[0][-4:].rjust(4, "0")
    elif len(parts) == 2:
        parts.insert(0, "0000")
    out = ":".join(parts)
    return out if "." in out else out + ".0"


def _csv_rows(text, fields):
    rows = []
    for rec in csv.reader(io.StringIO(text or ""), skipinitialspace=True):
        if not rec or not rec[0].strip():
            continue
        if len(rec) != len(fields):
            raise ValueError(f"expected {len(fields)} columns, got {len(rec)}: {rec}")
        rows.append({f: v.strip() for f, v in zip(fields, rec)})
    return rows


def gpu_csv(text):
    """nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits"""
    rows = _csv_rows(text, GPU_FIELDS)
    for r in rows:
        r["bus"] = norm_bus(r["pci.bus_id"])
    return rows


def remapped_csv(text):
    """nvidia-smi --query-remapped-rows=<REMAP_FIELDS> --format=csv,noheader"""
    rows = _csv_rows(text, REMAP_FIELDS)
    for r in rows:
        r["bus"] = norm_bus(r["gpu_bus_id"])
    return rows


def compute_apps(text):
    """nvidia-smi --query-compute-apps=<APP_FIELDS> --format=csv,noheader -> running GPU processes."""
    apps = []
    for line in (text or "").splitlines():
        rec = [x.strip() for x in line.split(",")]
        if rec and rec[0].isdigit():
            apps.append(dict(zip(APP_FIELDS, rec)))
    return apps


def as_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def smi_header(text):
    """Driver and CUDA version from the plain `nvidia-smi` banner."""
    drv = re.search(r"Driver Version:\s*([0-9.]+)", text or "")
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", text or "")
    return {"driver": drv.group(1) if drv else None, "cuda": cuda.group(1) if cuda else None}


def topo(text):
    """`nvidia-smi topo -m` -> {"matrix": {row: {col: value}}, "nics": {"NIC0": "mlx5_0", ...}}"""
    text = strip_ansi(text)
    lines = text.splitlines()
    header = None
    matrix = {}
    for line in lines:
        if header is None:
            cells = [c.strip() for c in line.split("\t") if c.strip()]
            if cells and cells[0].startswith("GPU"):
                header = cells
            continue
        if not line.strip() or line.startswith("Legend"):
            break
        cells = [c.strip() for c in line.split("\t")]
        label = cells[0]
        if not re.match(r"^(GPU|NIC)\d+$", label):
            continue
        matrix[label] = {col: (cells[i + 1] if i + 1 < len(cells) else "") for i, col in enumerate(header)}
    nics = dict(re.findall(r"^\s*(NIC\d+):\s*(\S+)\s*$", text, re.M))
    return {"matrix": matrix, "nics": nics}


RAIL_RANK = {"PIX": 0, "PXB": 1}


def rail_map(tp, rdma_devs):
    """GPU index -> nearest RDMA device at PIX/PXB distance (None if no NIC shares its PCIe switch)."""
    out = {}
    for label, row in tp["matrix"].items():
        if not label.startswith("GPU"):
            continue
        best = None
        for nic_label, dev in tp["nics"].items():
            if dev not in rdma_devs:
                continue
            rank = RAIL_RANK.get(row.get(nic_label, ""))
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, dev)
        out[int(label[3:])] = best[1] if best else None
    return out


def nvlink_status(text):
    """`nvidia-smi nvlink -s` -> {gpu index: {"active": n, "inactive": n}}"""
    out, gpu = {}, None
    for line in (text or "").splitlines():
        m = re.match(r"^GPU (\d+):", line)
        if m:
            gpu = int(m.group(1))
            out[gpu] = {"active": 0, "inactive": 0}
            continue
        m = re.match(r"^\s+Link (\d+):\s*(.*)$", line)
        if m and gpu is not None:
            out[gpu]["active" if "GB/s" in m.group(2) else "inactive"] += 1
    return out


def nvlink_errors(text):
    """`nvidia-smi nvlink -e` -> {(gpu, link, counter name): value}"""
    out, gpu = {}, None
    for line in (text or "").splitlines():
        m = re.match(r"^GPU (\d+):", line)
        if m:
            gpu = int(m.group(1))
            continue
        m = re.match(r"^\s+Link (\d+):\s*([A-Za-z][A-Za-z /-]*?):\s*(\d+)\s*$", line)
        if m and gpu is not None:
            out[(gpu, int(m.group(1)), m.group(2).strip())] = int(m.group(3))
    return out


def fabric(text):
    """`nvidia-smi -q` -> {bus: {"state": ..., "status": ...}} from each GPU's Fabric section."""
    out = {}
    chunks = re.split(r"^GPU ([0-9A-Fa-f]+:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])\s*$", text or "", flags=re.M)
    for bus, body in zip(chunks[1::2], chunks[2::2]):
        m = re.search(r"^\s*Fabric\s*$(.*?)(?=^[ ]{4}\S|\Z)", body, re.M | re.S)
        if not m:
            continue
        sec = m.group(1)
        st = re.search(r"^\s*State\s*:\s*(.+?)\s*$", sec, re.M)
        ss = re.search(r"^\s*Status\s*:\s*(.+?)\s*$", sec, re.M)
        out[norm_bus(bus)] = {"state": st.group(1) if st else None, "status": ss.group(1) if ss else None}
    return out


def lspci_count(text, device_id):
    """Count devices with [10de:<device_id>] in `lspci -nn -d 10de:` output."""
    return len(re.findall(r"\[10de:" + re.escape(device_id.lower()) + r"\]", (text or "").lower()))


def dcgmi_nvswitches(text):
    m = re.search(r"(\d+)\s+NvSwitch(?:es)?\s+found", text or "", re.I)
    return int(m.group(1)) if m else None


def dcgmi_version(text):
    m = re.search(r"version:\s*([0-9][0-9.]*)", text or "")
    return m.group(1) if m else None


DCGM_STATUS = {"pass": "pass", "skip": "skip", "warn": "warn", "fail": "fail"}


def dcgm_diag(text):
    """`dcgmi diag -r N -j` -> one row per (test, entity)."""
    doc = json.loads(text)
    root = doc.get("DCGM Diagnostic", doc)
    rows = []
    for cat in root.get("test_categories", []):
        for test in cat.get("tests", []):
            for res in test.get("results", []):
                msgs = []
                for key in ("warnings", "errors"):
                    for w in res.get(key) or []:
                        msgs.append(w.get("message", w.get("msg", str(w))) if isinstance(w, dict) else str(w))
                rows.append({
                    "category": cat.get("category"), "test": test.get("name"),
                    "group": res.get("entity_group", "GPU"), "entity": res.get("entity_id"),
                    "status": res.get("status"), "info": res.get("info") or [], "messages": msgs,
                })
    return rows, doc.get("metadata", {})


def dcgm_verdict(status):
    return DCGM_STATUS.get(str(status).lower(), "error")


def dcgm_pcie_bidir(info_lines):
    for line in info_lines:
        m = re.match(r"^\s*bidirectional bandwidth:\s*([0-9.]+)\s*GB/s", line)
        if m:
            return float(m.group(1))
    return None


def nccl_version(code):
    """nccl-tests prints nccl-library=22809 for 2.28.9 (major*10000 + minor*100 + patch)."""
    code = int(code)
    return f"{code // 10000}.{(code % 10000) // 100}.{code % 100}"


ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
                 r"\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def nccl_log(text):
    """nccl-tests *_perf output -> rows, version, GPU count and the out-of-bounds line."""
    text = text or ""
    lib = re.search(r"nccl-library=(\d+)", text)
    ngpus = re.search(r"nGpus\s+(\d+)", text)
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            g = m.groups()
            rows.append({"size": int(g[0]), "busbw": _f(g[7]), "wrong": g[8],
                         "busbw_ip": _f(g[11]), "wrong_ip": g[12]})
    return {
        "nccl_version": nccl_version(lib.group(1)) if lib else None,
        "ngpus": int(ngpus.group(1)) if ngpus else None,
        "rows": rows,
        "oob_ok": bool(re.search(r"Out of bounds values\s*:\s*0\s+OK", text)),
    }


def _f(v):
    try:
        return float(v)
    except ValueError:
        return None


def perftest_bw(text, size=65536):
    """ib_write_bw client output -> BW average (Gb/s) of the row for `size` bytes."""
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m and int(m.group(1)) == size:
            return float(m.group(4))
    return None


XID = re.compile(r"NVRM: Xid \((?:PCI:)?([0-9a-fA-F:.]+)\): (\d+)")


def xids(text):
    """Kernel log -> [(bus, xid, line)]"""
    out = []
    for line in (text or "").splitlines():
        m = XID.search(line)
        if m:
            out.append((norm_bus(m.group(1)), int(m.group(2)), line.strip()))
    return out


def mokutil(text):
    t = (text or "").lower()
    if "secureboot enabled" in t:
        return True
    if "secureboot disabled" in t:
        return False
    return None


def lspci_acs(text):
    """`lspci -D -vvv` -> {bdf: {"SrcValid": True, "ReqRedir": False, ...}} for devices with ACSCtl.
    Also returns the set of devices whose capabilities were hidden (lspci run without root)."""
    out, denied, cur = {}, set(), None
    for line in (text or "").splitlines():
        m = re.match(r"^([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\s", line)
        if m:
            cur = m.group(1).lower()
            continue
        if cur is None:
            continue
        if "Capabilities: <access denied>" in line:
            denied.add(cur)
        m = re.match(r"^\s+ACSCtl:\s*(.*)$", line)
        if m:
            out[cur] = {f[:-1]: f.endswith("+") for f in m.group(1).split() if f[-1:] in "+-"}
    return out, denied


def fio_gbs(text, rw):
    """fio --output-format=json -> bandwidth of the first job's `rw` ("read"/"write") in GB/s."""
    doc = json.loads(text)
    job = doc["jobs"][0][rw]
    return job["bw_bytes"] / 1e9


def ofed_version(text):
    line = (text or "").strip().splitlines()[:1]
    return line[0].rstrip(":") if line else None
