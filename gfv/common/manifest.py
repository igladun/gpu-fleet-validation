"""Node-class manifests: the expected hardware, software and performance of one node class.

Source files are TOML (manifests/*.toml). The single-file build converts them to JSON and bundles
them into the package, so a node needs no TOML parser; --manifest also accepts JSON."""

import json
import os

from .. import package_data

REQUIRED = {
    "gpus": int, "gpu_name": str, "driver": str, "pcie_width": int, "nvlinks_total": int,
    "nvswitches": int, "rdma_nics": int, "rdma_rate_gbps": int, "numa_nodes": int,
}
OPTIONAL = {
    "cuda": str, "nccl": str, "nvswitch_pci_device": str, "rdma_link_layer": str,
}
TABLES = ("k8s", "host", "firmware", "perf", "storage")


class ManifestError(Exception):
    pass


def _parse(text, name):
    if name.endswith(".toml"):
        try:
            import tomllib
        except ImportError as e:
            raise ManifestError(f"{name}: reading TOML needs Python 3.11+; use the built single file "
                                "(manifests are bundled as JSON) or pass a JSON manifest") from e
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ManifestError(f"{name}: {e}") from e
    try:
        return json.loads(text)
    except ValueError as e:
        raise ManifestError(f"{name}: {e}") from e


def _source_dir():
    """manifests/ next to the package when running from a repo checkout."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(here, "manifests")
    return d if os.path.isdir(d) else None


def bundled_classes():
    """{class name: (source label, document)} from the bundle, else from the checkout's manifests/."""
    found = {}
    index = package_data("manifests/index.json")
    if index:
        for fname in json.loads(index):
            doc = json.loads(package_data(f"manifests/{fname}"))
            for cls in doc.get("class", {}):
                found[cls] = (f"bundled:{fname}", doc)
        return found
    d = _source_dir()
    if d:
        for fname in sorted(os.listdir(d)):
            if fname.endswith((".toml", ".json")):
                with open(os.path.join(d, fname)) as f:
                    doc = _parse(f.read(), fname)
                for cls in doc.get("class", {}):
                    found[cls] = (os.path.join("manifests", fname), doc)
    return found


def load(path=None, cls=None):
    """Return (class name, class dict, source label, whole document)."""
    if path:
        try:
            with open(path) as f:
                doc = _parse(f.read(), path)
        except OSError as e:
            raise ManifestError(f"cannot read manifest {path}: {e}") from e
        classes = doc.get("class", {}) if isinstance(doc, dict) else None
        source = path
        if not isinstance(classes, dict):
            raise ManifestError(f"{path}: expected a [class.\"<name>\"] table")
        if not classes:
            raise ManifestError(f"{path}: no [class.\"<name>\"] table")
        if cls is None:
            if len(classes) != 1:
                raise ManifestError(f"{path} defines {sorted(classes)}; pick one with --class")
            cls = next(iter(classes))
    else:
        if cls is None:
            raise ManifestError("pass --class (bundled manifest) or --manifest FILE")
        bundled = bundled_classes()
        if cls not in bundled:
            raise ManifestError(f"no bundled class {cls!r}; known: {sorted(bundled) or 'none'}")
        source, doc = bundled[cls]
        classes = doc["class"]
    if cls not in classes:
        raise ManifestError(f"class {cls!r} not in {source}; known: {sorted(classes)}")
    spec = classes[cls]
    validate(cls, spec)
    return cls, spec, source, doc


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str_list(v):
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


# Keys of the tables L2 reads. A typo must not silently turn a check off.
PERF = {"nccl_allreduce_busbw_gbs_1node", "rdma_write_bw_gbps", "dcgm_pcie_bidir_gbs"}
HOST = {"kernel_cmdline_contains": _str_list, "pcie_acs": lambda v: v in ("disabled", "enabled"),
        "secure_boot": lambda v: isinstance(v, bool), "nvidia_peermem": lambda v: isinstance(v, bool),
        "power_profile": lambda v: isinstance(v, str)}
FIRMWARE = {"bios", "bmc", "nic", "gpu_vbios"}
STORAGE = {"path": lambda v: isinstance(v, str) and v.startswith("/"),
           "size_gib": lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0,
           "seq_write_gbs": _num, "seq_read_gbs": _num}


def validate(cls, spec):
    if not isinstance(spec, dict):
        raise ManifestError(f"class {cls!r} must be a table")
    problems = []
    for key, typ in REQUIRED.items():
        if key not in spec:
            problems.append(f"missing {key}")
        elif not isinstance(spec[key], typ) or isinstance(spec[key], bool):
            problems.append(f"{key} must be {typ.__name__}")
    for key, typ in OPTIONAL.items():
        if key in spec and not isinstance(spec[key], typ):
            problems.append(f"{key} must be {typ.__name__}")
    unknown = sorted(set(spec) - set(REQUIRED) - set(OPTIONAL) - set(TABLES))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    for t in TABLES:
        if t in spec and not isinstance(spec[t], dict):
            problems.append(f"[{t}] must be a table")

    def table(name):
        return spec.get(name) if isinstance(spec.get(name), dict) else {}

    for k, v in table("perf").items():
        if k not in PERF:
            problems.append(f"unknown perf.{k} (known: {', '.join(sorted(PERF))})")
        elif not _num(v):
            problems.append(f"perf.{k} must be a number")
    for k, v in table("host").items():
        if k not in HOST:
            problems.append(f"unknown host.{k}")
        elif not HOST[k](v):
            problems.append(f"host.{k} has an invalid value {v!r}")
    for k, v in table("firmware").items():
        if k not in FIRMWARE:
            problems.append(f"unknown firmware.{k}")
        elif not _str_list(v):
            problems.append(f"firmware.{k} must be a list of strings")
    storage = table("storage") if "storage" in spec else None
    if storage is not None:
        if "path" not in storage:
            problems.append("storage.path is required when [storage] is present")
        for k, v in storage.items():
            if k not in STORAGE:
                problems.append(f"unknown storage.{k}")
            elif not STORAGE[k](v):
                problems.append(f"storage.{k} has an invalid value {v!r}")
    if problems:
        raise ManifestError(f"class {cls!r}: " + "; ".join(problems))
