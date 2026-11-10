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
    """Version, commit and build date written by scripts/build-dist.sh; checkout runs report 'source'."""
    raw = package_data("_build.json")
    info = json.loads(raw) if raw else {"commit": "source", "built": None}
    info["version"] = __version__
    return info


def tools_lock():
    return json.loads(package_data("tools.lock.json"))
