"""Test helpers: build a fake node (fixtures + fake sysfs + fake tools) and run the real CLI against it."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
FIXTURES = os.path.join(TESTS, "fixtures")
REFERENCE = os.path.join(FIXTURES, "h100-8x-ib")
FAKEBIN = os.path.join(TESTS, "fakebin")
MANIFEST_JSON = os.path.join(FIXTURES, "manifest-reference.json")

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def read(name, base=REFERENCE):
    with open(os.path.join(base, name)) as f:
        return f.read()


def build_sysroot(spec, root):
    """Create the files and symlinks described by sysfs.json under root."""
    for path, content in spec["files"].items():
        p = os.path.join(root, path.lstrip("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content + ("" if content.endswith("\n") else "\n"))
    for link, target in spec["symlinks"].items():
        lp = os.path.join(root, link.lstrip("/"))
        tp = os.path.join(root, target.lstrip("/"))
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        os.makedirs(tp, exist_ok=True)
        os.symlink(os.path.relpath(tp, os.path.dirname(lp)), lp)
    for d in spec.get("dirs", []):
        os.makedirs(os.path.join(root, d.lstrip("/")), exist_ok=True)


class FakeNode:
    """A temp directory holding a scenario (copy of the reference fixtures), a fake root and tool state."""

    def __init__(self, tools=None):
        self.tmp = tempfile.mkdtemp(prefix="gfv-test-")
        self.scenario = os.path.join(self.tmp, "scenario")
        shutil.copytree(REFERENCE, self.scenario)
        self.root = os.path.join(self.tmp, "root")
        self.state = os.path.join(self.tmp, "state")
        self.out = os.path.join(self.tmp, "evidence")
        self.tools_dir = os.path.join(self.tmp, "tools")
        os.makedirs(self.state)
        self.spec = json.loads(read("sysfs.json"))
        self.bin = FAKEBIN
        if tools is not None:   # a fakebin with only some tools, to test missing ones
            self.bin = os.path.join(self.tmp, "bin")
            os.makedirs(self.bin)
            shutil.copy(os.path.join(FAKEBIN, "_shim.sh"), self.bin)
            shutil.copy(os.path.join(FAKEBIN, "fake_tool.py"), self.bin)
            for t in tools:
                os.symlink("_shim.sh", os.path.join(self.bin, t))

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- scenario edits ---------------------------------------------------------------------
    def path(self, name):
        return os.path.join(self.scenario, name)

    def text(self, name):
        with open(self.path(name)) as f:
            return f.read()

    def write(self, name, content):
        with open(self.path(name), "w") as f:
            f.write(content)

    def edit(self, name, old, new, count=1):
        t = self.text(name)
        assert old in t, f"{old!r} not in {name}"
        self.write(name, t.replace(old, new, count))

    def sysfs(self, path, content):
        self.spec["files"][path] = content

    def build(self):
        build_sysroot(self.spec, self.root)
        return self

    def env(self, **extra):
        env = {
            "PATH": self.bin, "GFV_FAKE_PYTHON": sys.executable, "GFV_FAKE_DIR": self.scenario,
            "GFV_FAKE_STATE": self.state, "GFV_SYSROOT": self.root, "GFV_ALLOW_NONROOT": "1",
            "GFV_SETTLE_SECONDS": "0", "PYTHONPATH": REPO, "HOME": self.tmp, "LANG": "C.UTF-8",
        }
        env.update(extra)
        return env

    def calls(self):
        p = os.path.join(self.state, "calls.log")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(line)[:-1] for line in f]

    def accept(self, *args, manifest=MANIFEST_JSON, env=None):
        cmd = [sys.executable, "-m", "gfv", "accept", "--manifest", manifest, "--node", "gpu-node-01",
               "--out", self.out, "--tools-dir", self.tools_dir, *args]
        p = subprocess.run(cmd, cwd=REPO, env=env or self.env(), capture_output=True, text=True, timeout=120)
        results = []
        rp = os.path.join(self.out, "results.jsonl")
        if os.path.exists(rp):
            with open(rp) as f:
                results = [json.loads(line) for line in f]
        return Run(p.returncode, p.stdout, p.stderr, results)


class Run:
    def __init__(self, rc, stdout, stderr, results):
        self.rc, self.stdout, self.stderr, self.results = rc, stdout, stderr, results

    def by(self, check, verdict=None):
        return [r for r in self.results if r["check"] == check and (verdict is None or r["verdict"] == verdict)]

    def failing(self):
        return sorted((r["check"], r["target"]) for r in self.results if r["verdict"] in ("fail", "error"))

    def verdicts(self, check):
        return sorted({r["verdict"] for r in self.results if r["check"] == check})
