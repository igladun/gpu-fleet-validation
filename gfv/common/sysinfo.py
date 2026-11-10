"""Read-only access to /sys, /proc and /etc through a root prefix, so tests can point at a fake tree."""

import glob
import os


class SysFS:
    def __init__(self, root="/"):
        self.root = root

    def path(self, p):
        return os.path.join(self.root, p.lstrip("/"))

    def read(self, p, default=None):
        try:
            with open(self.path(p), errors="replace") as f:
                return f.read().strip()
        except OSError:
            return default

    def exists(self, p):
        return os.path.exists(self.path(p))

    def listdir(self, p):
        try:
            return sorted(os.listdir(self.path(p)))
        except OSError:
            return []

    def glob(self, pattern):
        base = self.path("/")
        return sorted("/" + os.path.relpath(g, base) for g in glob.glob(self.path(pattern)))

    def realpath(self, p):
        """Resolve symlinks inside the prefix and return a path relative to the prefix ("/sys/...")."""
        real = os.path.realpath(self.path(p))
        base = os.path.realpath(self.path("/"))
        rel = os.path.relpath(real, base)
        return "/" + rel if not rel.startswith("..") else real

    def os_release(self):
        out = {}
        for line in (self.read("/etc/os-release", "") or "").splitlines():
            k, sep, v = line.partition("=")
            if sep:
                out[k.strip()] = v.strip().strip('"')
        return out
