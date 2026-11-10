"""gpu-fleet-validation: validation scripts for multi-node GPU training clusters."""

import json
import pkgutil

__version__ = "0.1.0"


def package_data(name):
    """Return the bytes of a file shipped inside the package (works from a zipapp too), or None."""
    try:
        return pkgutil.get_data("gfv", name)
    except (FileNotFoundError, OSError):
        return None


def build_info():
    """Version and source hash written by scripts/build-dist.sh; checkout runs report 'checkout'."""
    raw = package_data("_build.json")
    source = json.loads(raw).get("source") if raw else None
    return {"version": __version__, "source": source,
            "label": f"{__version__} (src {source})" if source else f"{__version__} (checkout)"}


def tools_lock():
    return json.loads(package_data("tools.lock.json"))
