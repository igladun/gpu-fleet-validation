"""Installer against fake apt/dpkg/git/make: which commands run, pins, idempotency, allowed packages."""

import io
import json
import os
import re
import sys
import tempfile
import unittest

from helpers import FIXTURES, FakeNode

from gfv import tools_lock
from gfv.acceptance import parse
from gfv.acceptance.context import Ctx
from gfv.common import install
from gfv.common.proc import Proc
from gfv.common.results import Recorder
from gfv.common.sysinfo import SysFS

ALLOWED = [r"ca-certificates", r"fio", r"pciutils", r"mokutil", r"git", r"build-essential", r"perftest",
           r"datacenter-gpu-manager-4-cuda\d+(=.+)?", r"cuda-nvcc-\d+-\d+", r"cuda-cudart-dev-\d+-\d+"]
INSTALL_TOOLS = ["apt-get", "apt-cache", "dpkg", "dpkg-deb", "git", "make", "nvidia-smi"]
MADISON = os.path.join(FIXTURES, "install", "apt-madison.txt")


class Args:
    pass


def fake_fetch(url, dest):
    with open(dest, "w") as f:
        f.write(url)


class InstallCase(unittest.TestCase):
    present = []          # extra tools that already exist on the node

    def setUp(self):
        self.node = FakeNode(tools=INSTALL_TOOLS + self.present)
        self.addCleanup(self.node.cleanup)
        with open(MADISON) as f:
            self.node.write("apt-madison.txt", f.read())
        self.node.build()

    def ctx(self, cuda="13.0", nccl="2.28.9"):
        ca = os.path.join(self.node.root, "etc/ssl/certs/ca-certificates.crt")
        if not os.path.exists(ca) and not getattr(self, "no_ca", False):
            os.makedirs(os.path.dirname(ca), exist_ok=True)
            open(ca, "w").close()
        out = tempfile.mkdtemp(dir=self.node.tmp)
        env = self.node.env()
        rec = Recorder({"run_id": "t", "tier": "L2"}, out, log=io.StringIO())
        spec = {"nccl": nccl} if nccl else {}
        ctx = Ctx(Args(), "c", spec, out, Proc(out, env=env), SysFS(self.node.root), rec, settle=0)
        ctx.facts = {"cuda": cuda, "gpus": parse.gpu_csv(self.node.text("gpus.csv"))}
        return ctx

    def install(self, ctx):
        return install.run(ctx, self.node.tools_dir, fetch=fake_fetch, machine="x86_64")

    def apt_installs(self):
        pkgs = []
        for c in self.node.calls():
            if c[:2] == ["apt-get", "install"]:
                pkgs += [a for a in c[2:] if not a.startswith("-")]
        return pkgs


class TestFreshNode(InstallCase):
    def test_fresh_install_sequence(self):
        ctx = self.ctx()
        self.install(ctx)
        calls = self.node.calls()
        tools = [c[0] for c in calls]
        self.assertEqual(tools.count("dpkg"), 1, "cuda-keyring added once")
        self.assertEqual([c for c in calls if c[:2] == ["apt-get", "update"]], [["apt-get", "update"]])
        lock = tools_lock()
        self.assertIn(["apt-get", "install", "-y", "--install-recommends",
                       f"datacenter-gpu-manager-4-cuda13={lock['dcgm']['version']}"], calls)
        self.assertIn(["apt-get", "download", "libnccl2=2.28.9-1+cuda13.0", "libnccl-dev=2.28.9-1+cuda13.0"], calls)
        self.assertIn(["apt-get", "install", "-y", "--no-install-recommends", "cuda-nvcc-13-0",
                       "cuda-cudart-dev-13-0"], calls)
        fetch = next(c for c in calls if c[0] == "git" and "fetch" in c)
        self.assertEqual(fetch[-2:], [lock["nccl_tests"]["repo"], lock["nccl_tests"]["commit"]])
        make = next(c for c in calls if c[0] == "make")
        self.assertIn("NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90", make)
        nccl_home = next(a for a in make if a.startswith("NCCL_HOME="))[len("NCCL_HOME="):]
        self.assertTrue(os.path.exists(os.path.join(nccl_home, "include", "nccl.h")))
        self.assertTrue(os.path.exists(os.path.join(nccl_home, "lib", "libnccl.so.2")))
        self.assertTrue(ctx.nccl_tests and os.access(os.path.join(ctx.nccl_tests, "build", "all_reduce_perf"), os.X_OK))
        self.assertEqual(ctx.ld_path, [os.path.join(nccl_home, "lib")])
        self.assertEqual([r for r in ctx.rec.results if r["verdict"] == "error"], [])
        with open(os.path.join(ctx.out_dir, "installed.json")) as f:
            done = json.load(f)
        self.assertEqual(done["nccl"]["version"], "2.28.9-1+cuda13.0")

    def test_only_allowed_packages(self):
        self.install(self.ctx())
        pkgs = self.apt_installs()
        self.assertTrue(pkgs)
        for p in pkgs:
            self.assertTrue(any(re.fullmatch(a, p) for a in ALLOWED), p)
        touched = {c[0] for c in self.node.calls()}
        self.assertNotIn("apt-mark", touched)
        self.assertFalse(any("upgrade" in c for c in self.node.calls()))
        self.assertFalse(any(re.search(r"nvidia-driver|fabricmanager|doca|ofed|linux-", " ".join(c))
                             for c in self.node.calls() if c[0] == "apt-get"))

    def test_rerun_is_idempotent_for_nccl_tests(self):
        self.install(self.ctx())
        os.remove(os.path.join(self.node.state, "calls.log"))
        ctx = self.ctx()
        self.install(ctx)
        calls = self.node.calls()
        self.assertFalse(any(c[0] in ("git", "make", "dpkg-deb", "apt-cache") for c in calls), calls)
        self.assertFalse(any(c[:2] == ["apt-get", "download"] for c in calls))
        self.assertEqual(ctx.rec.results[-1]["check"], "install.nccl_tests")
        self.assertEqual(ctx.rec.results[-1]["detail"], "already built")

    def test_dcgm_pin_falls_back_to_unpinned(self):
        self.node.write("apt-install-fail", "datacenter-gpu-manager-4-cuda13=")
        ctx = self.ctx()
        self.install(ctx)
        self.assertIn(["apt-get", "install", "-y", "--install-recommends", "datacenter-gpu-manager-4-cuda13"],
                      self.node.calls())

    def test_no_nccl_for_this_cuda(self):
        ctx = self.ctx(cuda="13.0", nccl="2.99.0")
        self.install(ctx)
        err = [r for r in ctx.rec.results if r["verdict"] == "error"]
        self.assertEqual([r["check"] for r in err], ["install.nccl"])
        self.assertIsNone(ctx.nccl_tests)

    def test_unsupported_distro(self):
        self.node.sysfs("/etc/os-release", 'ID=rocky\nPRETTY_NAME="Rocky Linux 9"\nVERSION_ID="9.4"')
        self.node.root += "-rocky"
        self.node.build()
        ctx = self.ctx()
        self.install(ctx)
        self.assertEqual([(r["check"], r["verdict"]) for r in ctx.rec.results], [("install", "error")])
        self.assertFalse(any(c[0] == "apt-get" for c in self.node.calls()))


class TestMinimalImage(InstallCase):
    no_ca = True

    def test_ca_bundle_installed_first(self):
        self.install(self.ctx())
        installs = [c for c in self.node.calls() if c[:2] == ["apt-get", "install"]]
        self.assertEqual(installs[0][-1], "ca-certificates")


class TestToolsPresent(InstallCase):
    present = ["dcgmi", "fio", "lspci", "mokutil", "ib_write_bw"]

    def test_present_tools_not_reinstalled(self):
        os.makedirs(os.path.join(self.node.root, "usr/local/cuda/bin"))
        open(os.path.join(self.node.root, "usr/local/cuda/bin/nvcc"), "w").close()
        ctx = self.ctx()
        self.install(ctx)
        pkgs = self.apt_installs()
        self.assertFalse(any(p.startswith("datacenter-gpu-manager") for p in pkgs), pkgs)
        self.assertFalse(any(p.startswith("cuda-nvcc") for p in pkgs), pkgs)
        self.assertEqual(set(pkgs), {"build-essential"})      # git/make/g++ shims exist, g++ does not


class TestPickNccl(unittest.TestCase):
    def setUp(self):
        with open(MADISON) as f:
            self.versions = [line.split("|")[1].strip() for line in f]

    def test_exact(self):
        self.assertEqual(install.pick_nccl(self.versions, "2.28.9", "13.0"), "2.28.9-1+cuda13.0")

    def test_newest_for_driver_cuda(self):
        self.assertEqual(install.pick_nccl(self.versions, None, "13.0"), "2.28.9-1+cuda13.0")
        self.assertEqual(install.pick_nccl(self.versions, None, "13.4"), "2.32.3-1+cuda13.4")

    def test_never_newer_cuda_minor_than_driver(self):
        self.assertIsNone(install.pick_nccl(self.versions, "2.32.3", "13.0"))

    def test_older_cuda_minor_allowed(self):
        self.assertEqual(install.pick_nccl(self.versions, "2.28.9", "13.2"), "2.28.9-1+cuda13.0")

    def test_bad_cuda(self):
        self.assertIsNone(install.pick_nccl(self.versions, None, None))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
