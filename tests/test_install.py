"""Installer against fake apt/dpkg/git/make: which commands run, pins, idempotency, allowed packages,
and the host-mutation guard (simulation refusals, keyring hash, repo removal)."""

import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

from helpers import FIXTURES, FakeNode

from gfv import tools_lock
from gfv.acceptance import parse
from gfv.acceptance.context import Ctx
from gfv.common import install
from gfv.common.proc import Proc
from gfv.common.results import Recorder
from gfv.common.sysinfo import SysFS

ALLOWED = [r"ca-certificates", r"fio", r"pciutils", r"mokutil", r"git", r"build-essential", r"perftest", r"libnuma1",
           r"datacenter-gpu-manager-4-(proprietary-)?cuda\d+(=.+)?", r"datacenter-gpu-manager-4-proprietary(=.+)?",
           r"cuda-nvcc-\d+-\d+", r"cuda-cudart-dev-\d+-\d+"]
INSTALL_TOOLS = ["apt-get", "apt-cache", "dpkg", "dpkg-deb", "dpkg-query", "git", "make", "nvidia-smi"]
MADISON = os.path.join(FIXTURES, "install", "apt-madison.txt")
KEYRING = b"fake cuda-keyring deb"
LOCK = tools_lock()
DCGM_PINNED = [f"{p.format(cuda_major='13')}={LOCK['dcgm']['version']}" for p in LOCK["dcgm"]["packages"]] + \
    LOCK["dcgm"]["extra"]


class Args:
    pass


def fake_fetch(url, dest):
    with open(dest, "wb") as f:
        f.write(KEYRING)


def test_lock():
    lock = copy.deepcopy(LOCK)
    for k in lock["cuda_repo"]["keyring_sha256"]:
        lock["cuda_repo"]["keyring_sha256"][k] = hashlib.sha256(KEYRING).hexdigest()
    return lock


class InstallCase(unittest.TestCase):
    present = []          # extra tools that already exist on the node
    no_ca = False

    def setUp(self):
        self.node = FakeNode(tools=INSTALL_TOOLS + self.present)
        self.addCleanup(self.node.cleanup)
        with open(MADISON) as f:
            self.node.write("apt-madison.txt", f.read())
        self.node.build()
        patcher = mock.patch.object(install, "tools_lock", test_lock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ctx(self, cuda="13.0", nccl="2.28.9"):
        ca = os.path.join(self.node.root, "etc/ssl/certs/ca-certificates.crt")
        if not os.path.exists(ca) and not self.no_ca:
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

    def errors(self, ctx):
        return [(r["check"], r["detail"]) for r in ctx.rec.results if r["verdict"] == "error"]


class TestFreshNode(InstallCase):
    def test_fresh_install_sequence(self):
        ctx = self.install_and_check()
        calls = self.node.calls()
        self.assertEqual([c[:2] for c in calls if c[0] == "dpkg"], [["dpkg", "-i"], ["dpkg", "--purge"]])
        self.assertEqual([c for c in calls if c[:2] == ["apt-get", "update"]], [["apt-get", "update"]])
        self.assertIn(["apt-get", "install", "-y", "--no-remove", "--no-install-recommends", *DCGM_PINNED], calls)
        self.assertIn(["apt-get", "download", "libnccl2=2.28.9-1+cuda13.0", "libnccl-dev=2.28.9-1+cuda13.0"], calls)
        self.assertIn(["apt-get", "install", "-y", "--no-remove", "--no-install-recommends", "cuda-nvcc-13-0",
                       "cuda-cudart-dev-13-0"], calls)
        fetch = next(c for c in calls if c[0] == "git" and "fetch" in c)
        self.assertEqual(fetch[-2:], [LOCK["nccl_tests"]["repo"], LOCK["nccl_tests"]["commit"]])
        make = next(c for c in calls if c[0] == "make")
        self.assertIn("NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90", make)
        nccl_home = next(a for a in make if a.startswith("NCCL_HOME="))[len("NCCL_HOME="):]
        self.assertTrue(os.path.exists(os.path.join(nccl_home, "include", "nccl.h")))
        self.assertTrue(os.path.exists(os.path.join(nccl_home, "lib", "libnccl.so.2")))
        self.assertTrue(ctx.nccl_tests and os.access(os.path.join(ctx.nccl_tests, "build", "all_reduce_perf"), os.X_OK))
        self.assertEqual(ctx.ld_path, [os.path.join(nccl_home, "lib")])
        with open(os.path.join(ctx.out_dir, "installed.json")) as f:
            done = json.load(f)
        self.assertEqual(done["nccl"]["version"], "2.28.9-1+cuda13.0")
        self.assertTrue(done["cuda_keyring_removed"])
        self.assertFalse(os.path.exists(os.path.join(self.node.root, "etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list")))

    def install_and_check(self):
        ctx = self.ctx()
        self.install(ctx)
        self.assertEqual(self.errors(ctx), [])
        return ctx

    def test_every_install_is_simulated_first(self):
        self.install_and_check()
        calls = self.node.calls()
        for i, c in enumerate(calls):
            if c[:2] == ["apt-get", "install"]:
                self.assertEqual(calls[i - 1][:2], ["apt-get", "-s"], c)
                self.assertIn("--no-remove", c)

    def test_only_allowed_packages(self):
        self.install_and_check()
        pkgs = self.apt_installs()
        self.assertTrue(pkgs)
        for p in pkgs:
            self.assertTrue(any(re.fullmatch(a, p) for a in ALLOWED), p)
        self.assertNotIn("apt-mark", {c[0] for c in self.node.calls()})
        self.assertFalse(any("upgrade" in c for c in self.node.calls()))

    def test_rerun_is_idempotent_for_nccl_tests(self):
        self.install_and_check()
        os.remove(os.path.join(self.node.state, "calls.log"))
        ctx = self.ctx()
        self.install(ctx)
        calls = self.node.calls()
        self.assertFalse(any(c[0] in ("git", "make", "dpkg-deb", "apt-cache") for c in calls), calls)
        self.assertFalse(any(c[:2] == ["apt-get", "download"] for c in calls))
        built = [r for r in ctx.rec.results if r["check"] == "install.nccl_tests"]
        self.assertEqual([r["detail"] for r in built], ["already built"])

    def test_dcgm_pin_falls_back_to_unpinned(self):
        self.node.write("apt-install-fail", "datacenter-gpu-manager-4-cuda13=")
        self.install(self.ctx())
        unpinned = [p.format(cuda_major="13") for p in LOCK["dcgm"]["packages"]] + LOCK["dcgm"]["extra"]
        self.assertIn(["apt-get", "install", "-y", "--no-remove", "--no-install-recommends", *unpinned],
                      self.node.calls())

    def test_no_nccl_for_this_cuda(self):
        ctx = self.ctx(cuda="13.0", nccl="2.99.0")
        self.install(ctx)
        self.assertEqual([e[0] for e in self.errors(ctx)], ["install.nccl"])
        self.assertIsNone(ctx.nccl_tests)

    def test_unsupported_distro(self):
        self.node.sysfs("/etc/os-release", 'ID=rocky\nPRETTY_NAME="Rocky Linux 9"\nVERSION_ID="9.4"')
        self.node.root += "-rocky"
        self.node.build()
        ctx = self.ctx()
        self.install(ctx)
        self.assertEqual([(r["check"], r["verdict"]) for r in ctx.rec.results], [("install", "error")])
        self.assertFalse(any(c[0] == "apt-get" for c in self.node.calls()))


class TestHostGuard(InstallCase):
    def test_dependency_upgrades_allowed_and_recorded(self):
        self.node.write("apt-sim.txt", "Inst perl-base [5.38.2-3.2ubuntu0.6] (5.38.2-3.2ubuntu0.7 Ubuntu:24.04/noble-security [amd64])\n"
                                       "Inst perl (5.38.2-3.2ubuntu0.7 Ubuntu:24.04/noble-security [amd64])\n")
        ctx = self.ctx()
        self.install(ctx)
        self.assertEqual(self.errors(ctx), [])
        up = [r for r in ctx.rec.results if r["check"] == "install.upgraded"]
        self.assertTrue(up and all(r["verdict"] == "warn" for r in up))
        self.assertIn("perl-base 5.38.2-3.2ubuntu0.6 -> 5.38.2-3.2ubuntu0.7", up[0]["value"])
        with open(os.path.join(ctx.out_dir, "installed.json")) as f:
            self.assertTrue(json.load(f)["upgraded"])

    def test_refuses_upgrading_the_stack_under_test(self):
        self.node.write("apt-sim.txt", "Inst libibverbs1 [2501.0-1] (50.0-2build2 Ubuntu:24.04/noble [amd64])\n"
                                       "Inst perftest (24.10.0-0.66 Ubuntu:24.04/noble [amd64])\n")
        ctx = self.ctx()
        self.install(ctx)
        errs = self.errors(ctx)
        self.assertIn("install.apt_tools", [e[0] for e in errs])
        self.assertTrue(any("would upgrade libibverbs1" in e[1] for e in errs), errs)
        self.assertEqual([c for c in self.node.calls() if c[:2] == ["apt-get", "install"]], [])

    def test_refuses_removals_and_protected_packages(self):
        self.node.write("apt-sim.txt", "Remv rdma-core [2501.0]\nInst nvidia-driver-580 (580.95.05-0ubuntu1 [amd64])\n")
        ctx = self.ctx()
        self.install(ctx)
        detail = " ".join(e[1] for e in self.errors(ctx))
        self.assertIn("would remove rdma-core", detail)
        self.assertIn("would install protected package nvidia-driver-580", detail)
        self.assertEqual([c for c in self.node.calls() if c[:2] == ["apt-get", "install"]], [])

    def test_new_rdma_dependencies_are_flagged(self):
        self.node.write("apt-sim.txt", "Inst libibverbs1 (50.0-2build2 Ubuntu:24.04/noble [amd64])\n"
                                       "Inst perftest (24.10.0-0.66 Ubuntu:24.04/noble [amd64])\n")
        ctx = self.ctx()
        self.install(ctx)
        dep = [r for r in ctx.rec.results if r["check"] == "install.new_dependencies"]
        self.assertTrue(dep and dep[0]["verdict"] == "warn" and "libibverbs1" in dep[0]["value"])

    def test_operator_keyring_without_source_is_not_touched(self):
        self.node.write("keyring-installed", "")
        ctx = self.ctx()
        self.install(ctx)
        self.assertIn("install.cuda_repo", [e[0] for e in self.errors(ctx)])
        self.assertFalse(any(c[0] == "dpkg" for c in self.node.calls()))

    def test_needrestart_never_restarts(self):
        self.assertEqual(install.APT_ENV["NEEDRESTART_MODE"], "l")

    def test_keyring_hash_mismatch(self):
        def tampered(url, dest):
            with open(dest, "wb") as f:
                f.write(b"tampered")
        ctx = self.ctx()
        install.run(ctx, self.node.tools_dir, fetch=tampered, machine="x86_64")
        self.assertIn(("install.cuda_repo", "cuda-keyring sha256 mismatch"), self.errors(ctx))
        self.assertFalse(any(c[0] == "dpkg" for c in self.node.calls()))

    def test_existing_repo_is_left_alone(self):
        d = os.path.join(self.node.root, "etc/apt/sources.list.d")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "cuda.list"), "w") as f:
            f.write("deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /\n")
        self.install(self.ctx())
        self.assertFalse(any(c[0] == "dpkg" for c in self.node.calls()))


class TestMinimalImage(InstallCase):
    no_ca = True

    def test_ca_bundle_installed_first(self):
        self.install(self.ctx())
        installs = [c for c in self.node.calls() if c[:2] == ["apt-get", "install"]]
        self.assertEqual(installs[0][-1], "ca-certificates")


class TestToolsPresent(InstallCase):
    present = ["dcgmi", "fio", "lspci", "mokutil", "ib_write_bw"]

    def cuda(self, home, version):
        os.makedirs(os.path.join(self.node.root, home.lstrip("/"), "bin"))
        open(os.path.join(self.node.root, home.lstrip("/"), "bin/nvcc"), "w").close()
        with open(os.path.join(self.node.root, home.lstrip("/"), "version.json"), "w") as f:
            json.dump({"cuda": {"name": "CUDA SDK", "version": version}}, f)

    def test_present_tools_not_reinstalled(self):
        self.cuda("/usr/local/cuda", "13.0.48")
        ctx = self.ctx()
        self.install(ctx)
        pkgs = self.apt_installs()
        self.assertFalse(any(p.startswith("datacenter-gpu-manager") for p in pkgs), pkgs)
        self.assertFalse(any(p.startswith("cuda-nvcc") for p in pkgs), pkgs)
        self.assertEqual(set(pkgs), {"build-essential"})      # git/make/g++ shims exist, g++ does not

    def test_toolkit_of_another_major_is_not_used(self):
        self.cuda("/usr/local/cuda", "12.4.131")
        self.install(self.ctx())
        self.assertIn("cuda-nvcc-13-0", self.apt_installs())


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


class TestSimulationParser(unittest.TestCase):
    def test_new_packages_only_is_clean(self):
        self.assertEqual(install.apt_simulation_problems("Inst fio (3.36-1 Ubuntu:24.04/noble [amd64])\n"
                                                         "Conf fio (3.36-1 Ubuntu:24.04/noble [amd64])\n"),
                         ([], [], ["fio"]))

    def test_protected_new_dependencies_refused(self):
        for pkg in ("nvidia-utils-580", "nvidia-compute-utils-580", "nvidia-imex-580", "cuda-compat-13-0", "dkms",
                    "libnccl2", "linux-modules-extra-6.8.0-90-generic"):
            problems, _, _ = install.apt_simulation_problems(f"Inst {pkg} (1.0 x [amd64])\n")
            self.assertTrue(problems, pkg)
        for pkg in ("rdma-core", "ibverbs-providers", "libnuma1", "cuda-crt-13-0", "datacenter-gpu-manager-4-core",
                    "linux-libc-dev", "cuda-driver-dev-13-0", "libxnvctrl0"):
            problems, _, new = install.apt_simulation_problems(f"Inst {pkg} (1.0 x [amd64])\n")
            self.assertEqual((problems, new), ([], [pkg]))

    def test_driver_and_kernel_upgrades_refused(self):
        for pkg in ("nvidia-fabricmanager", "libnvidia-compute-580", "linux-image-6.8.0-90-generic", "libnccl2",
                    "rdma-core", "doca-ofed", "linux-modules-6.8.0-90-generic", "linux-generic"):
            problems, _, _ = install.apt_simulation_problems(f"Inst {pkg} [1.0] (1.1 x [amd64])\n")
            self.assertTrue(problems, pkg)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
