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
        classes = doc.get("class", {})
        source = path
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


def validate(cls, spec):
    problems = []
    for key, typ in REQUIRED.items():
        if key not in spec:
            problems.append(f"missing {key}")
        elif not isinstance(spec[key], typ) or isinstance(spec[key], bool):
            problems.append(f"{key} must be {typ.__name__}")
    for key, typ in OPTIONAL.items():
        if key in spec and not isinstance(spec[key], typ):
            problems.append(f"{key} must be {typ.__name__}")
    for t in TABLES:
        if t in spec and not isinstance(spec[t], dict):
            problems.append(f"[{t}] must be a table")
    for k, v in spec.get("perf", {}).items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            problems.append(f"perf.{k} must be a number")
    for k, v in spec.get("firmware", {}).items():
        if not isinstance(v, list):
            problems.append(f"firmware.{k} must be a list")
    storage = spec.get("storage")
    if storage is not None and "path" not in storage:
        problems.append("storage.path is required when [storage] is present")
    if problems:
        raise ManifestError(f"class {cls!r}: " + "; ".join(problems))
