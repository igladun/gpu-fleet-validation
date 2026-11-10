"""Install the test tools a run needs. Idempotent, test tooling only.

Never touches the GPU driver, Fabric Manager, OFED/DOCA, the kernel or package holds: every apt
transaction is simulated first and refused if it would remove anything, pull in a driver/kernel package,
or upgrade any part of the stack under test (NVIDIA, CUDA driver, OFED/DOCA, RDMA userspace, system NCCL).
Other dependency upgrades (e.g. a patch-level perl or libc bump) are allowed and recorded as a warning.
needrestart restarts nothing. NCCL is unpacked into the tools dir with dpkg-deb
(the system NCCL is left alone); nccl-tests is built there at a pinned commit. A CUDA apt repo added for
the install is removed again at the end (its keyring ships an apt pin that would otherwise prefer
NVIDIA's packages on later upgrades). Everything done is written to installed.json in the evidence dir."""

import glob
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import urllib.request

from .. import tools_lock

APT_ENV = {"DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_MODE": "l", "NEEDRESTART_SUSPEND": "1"}
# Never pulled in as a new package, even as a dependency: driver, kernel, CUDA driver/compat, system NCCL,
# OFED/DOCA. (A new nvidia-utils-* would overwrite the nvidia-smi of a .run-installed driver.)
# linux-* means kernel packages; linux-libc-dev (userspace headers for libc6-dev) and linux-base are not.
KERNEL = r"linux-(?!libc-dev$|base$)"
PROTECTED_NEW = re.compile(rf"^(nvidia-|libnvidia-|cuda-drivers|cuda-compat|dkms$|{KERNEL}|libnccl|mlnx|doca|ofed|knem)")
# New RDMA userspace is allowed (perftest needs it on inbox-driver hosts) but flagged.
RDMA_USERSPACE = re.compile(r"^(rdma-core|libibverbs|ibverbs-|librdmacm|libibumad|infiniband-diags)")
# Never upgraded: the stack the gate is testing.
PROTECTED_UPGRADE = re.compile(rf"^(nvidia-|libnvidia-|cuda-drivers|cuda-compat|{KERNEL}|mlnx|doca|ofed|knem|rdma-core"
                               r"|libibverbs|ibverbs-|librdmacm|libibumad|infiniband-diags|libnccl|datacenter-gpu-manager)")
CUDA_REPO_MARK = "developer.download.nvidia.com/compute/cuda/repos"


def _fetch(url, dest, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


class Installer:
    def __init__(self, ctx, tools_dir, fetch=_fetch, machine=None):
        self.ctx, self.tools_dir, self.fetch = ctx, tools_dir, fetch
        self.lock = tools_lock()
        self.machine = machine or platform.machine()
        self.updated = False
        self.done = {"apt_packages": [], "cuda_keyring": None, "dcgm": None, "nccl": None, "nvcc": None,
                     "nccl_tests": None}

    # --- helpers ----------------------------------------------------------------------------
    def _apt(self, name, args, timeout=1800):
        return self.ctx.proc.run(name, ["apt-get", *args], timeout=timeout,
                                 env_extra={**APT_ENV, **self.ctx.env_extra()})

    def _update(self):
        if not self.updated:
            r = self._apt("install-apt-update", ["update"], timeout=900)
            self.updated = r.ok
            return r
        return None

    def _install(self, name, pkgs):
        """Simulate, refuse anything beyond adding new test packages, then install with --no-remove."""
        self._update()
        args = ["install", "--no-remove", "--no-install-recommends", *pkgs]
        sim = self._apt(f"{name}-simulate", ["-s", *args])
        if not sim.ok:
            return sim
        problems, upgrades, new = apt_simulation_problems(sim.out)
        if problems:
            sim.rc = 1
            sim.err = "refused: " + "; ".join(problems[:8])
            return sim
        r = self._apt(name, [args[0], "-y", *args[1:]])
        if not r.ok:
            return r
        requested = {p.split("=")[0] for p in pkgs}
        deps = sorted(p for p in new if p not in requested)
        if deps:
            self.done.setdefault("new_dependencies", []).extend(deps)
            rdma = [p for p in deps if RDMA_USERSPACE.match(p)]
            self.ctx.add("install.new_dependencies", "warn" if rdma else "info", None, deps, None,
                         ("new RDMA userspace packages installed: " + ", ".join(rdma)) if rdma
                         else f"pulled in by {', '.join(sorted(requested))[:120]}", evidence=f"raw/{sim.name}.out")
        if upgrades:
            self.done.setdefault("upgraded", []).extend(upgrades)
            self.ctx.add("install.upgraded", "warn", None, upgrades, None,
                         f"apt upgraded these installed packages as dependencies of {', '.join(pkgs)[:120]}",
                         evidence=f"raw/{sim.name}.out")
        return r

    def repo(self):
        rel = self.ctx.sys.os_release()
        arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "sbsa", "arm64": "sbsa"}.get(self.machine)
        return "ubuntu" + rel.get("VERSION_ID", "").replace(".", ""), arch

    def cuda_repo_configured(self):
        files = ["/etc/apt/sources.list"] + self.ctx.sys.glob("/etc/apt/sources.list.d/*")
        return any(CUDA_REPO_MARK in (self.ctx.sys.read(f, "") or "") for f in files)

    # --- steps ------------------------------------------------------------------------------
    def supported(self):
        rel = self.ctx.sys.os_release()
        repo, arch = self.repo()
        if rel.get("ID") != "ubuntu" or not self.ctx.which("apt-get") or not arch:
            self.ctx.add("install", "error", None, rel.get("PRETTY_NAME"), "Ubuntu with apt-get",
                         "automatic install supports Ubuntu only; install the tools yourself and pass --no-install")
            return False
        return True

    def ensure_ca(self):
        """HTTPS for the keyring download and git needs a CA bundle; minimal images lack one."""
        if self.ctx.sys.exists("/etc/ssl/certs/ca-certificates.crt"):
            return
        r = self._install("install-ca-certificates", ["ca-certificates"])
        if r.ok:
            self.done["apt_packages"].append("ca-certificates")
            self.ctx.add("install.ca_certificates", "pass", None, "ca-certificates", None, "installed")
        else:
            self.ctx.add("install.ca_certificates", "error", None, "ca-certificates", None, r.why())

    def ensure_cuda_repo(self):
        if self.cuda_repo_configured():
            return True
        kr = self.ctx.proc.run("install-cuda-keyring-status", ["dpkg-query", "-W", "-f", "${db:Status-Abbrev}",
                                                              "cuda-keyring"], timeout=60)
        if kr.ok and kr.out.startswith("ii"):
            self.ctx.add("install.cuda_repo", "error", None, "cuda-keyring installed", "CUDA apt source",
                         "cuda-keyring is installed but its apt source is missing; fix /etc/apt or pass --no-install")
            return False
        repo, arch = self.repo()
        base = self.lock["cuda_repo"]["base_url"]
        deb = f"cuda-keyring_{self.lock['cuda_repo']['keyring_version']}_all.deb"
        url = f"{base}/{repo}/{arch}/{deb}"
        want = self.lock["cuda_repo"]["keyring_sha256"].get(f"{repo}/{arch}")
        if not want:
            self.ctx.add("install.cuda_repo", "error", None, f"{repo}/{arch}", None, "no pinned keyring hash")
            return False
        tmp = tempfile.mkdtemp(prefix="gfv-keyring-")
        try:
            path = os.path.join(tmp, deb)
            try:
                self.fetch(url, path)
            except OSError as e:
                self.ctx.add("install.cuda_repo", "error", None, None, url, f"download failed: {e}")
                return False
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != want:
                self.ctx.add("install.cuda_repo", "error", None, got, want, "cuda-keyring sha256 mismatch")
                return False
            r = self.ctx.proc.run("install-cuda-keyring", ["dpkg", "-i", path], timeout=300)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not r.ok:
            self.ctx.add("install.cuda_repo", "error", None, None, url, r.why())
            return False
        self.updated = False
        self.done["cuda_keyring"] = url
        self.ctx.add("install.cuda_repo", "pass", None, f"{repo}/{arch}", None,
                     "added NVIDIA CUDA apt repo for this install (removed again at the end)")
        return True

    def remove_cuda_repo(self):
        """Purge the keyring package this run added: its sources entry and its pin-600 preferences file."""
        if not self.done["cuda_keyring"]:
            return
        r = self.ctx.proc.run("install-cuda-keyring-purge", ["dpkg", "--purge", "cuda-keyring"], timeout=300)
        if r.ok:
            self.done["cuda_keyring_removed"] = True
            self.ctx.add("install.cuda_repo_removed", "pass", None, "cuda-keyring", None,
                         "apt sources and preferences restored")
        else:
            self.ctx.add("install.cuda_repo_removed", "error", None, "cuda-keyring", None,
                         f"could not purge cuda-keyring; remove it by hand ({r.why()})")

    def apt_tools(self):
        missing = sorted({pkg for tool, pkg in self.lock["apt_tools"].items() if not self.ctx.which(tool)})
        if not missing:
            self.ctx.add("install.apt_tools", "info", None, "all present", None)
            return
        r = self._install("install-apt-tools", missing)
        if r.ok:
            self.done["apt_packages"] += missing
            self.ctx.add("install.apt_tools", "pass", None, missing, None, "installed")
        else:
            self.ctx.add("install.apt_tools", "error", None, missing, None, r.why())

    def dcgm(self, cuda_major):
        if self.ctx.which("dcgmi"):
            self.ctx.add("install.dcgm", "info", None, "present", None, "using the DCGM already on the node")
            return
        if not self.ensure_cuda_repo():
            return
        lock = self.lock["dcgm"]
        names = [p.format(cuda_major=cuda_major) for p in lock["packages"]]
        pkgs = [f"{n}={lock['version']}" for n in names] + lock.get("extra", [])
        r = self._install("install-dcgm", pkgs)
        if not r.ok and "refused" not in r.err:
            pkgs = names + lock.get("extra", [])
            r = self._install("install-dcgm-unpinned", pkgs)
        if r.ok:
            self.done["dcgm"] = pkgs
            self.done["apt_packages"] += pkgs
            self.ctx.add("install.dcgm", "pass", None, pkgs[0], None, "installed")
        else:
            self.ctx.add("install.dcgm", "error", None, pkgs[0], None, r.why())

    def nccl_tests(self, cuda, nccl_want, compute_caps, jobs=None):
        """Download NCCL for the driver's CUDA, find or install nvcc, build pinned nccl-tests."""
        lock = self.lock["nccl_tests"]
        reuse = self.existing_build(cuda, nccl_want)
        if reuse:
            self.done["nccl_tests"] = reuse
            self.ctx.add("install.nccl_tests", "info", None, os.path.basename(reuse["dir"]), None, "already built")
            return reuse
        if not self.ensure_cuda_repo():
            return None
        self._update()
        mad = self.ctx.proc.run("install-nccl-madison", ["apt-cache", "madison", "libnccl2"], timeout=120)
        versions = [p.split("|")[1].strip() for p in mad.out.splitlines() if p.count("|") >= 2]
        pick = pick_nccl(versions, nccl_want, cuda)
        if not pick:
            self.ctx.add("install.nccl", "error", None, None, f"{nccl_want or 'any'} for CUDA {cuda}",
                         f"no libnccl2 build in the apt repo (found {len(versions)} versions)")
            return None
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", pick)
        tests_dir = os.path.join(self.tools_dir, f"nccl-tests-{lock['commit'][:12]}-nccl-{safe}")
        meta_path = os.path.join(tests_dir, "gfv-tool.json")
        nccl_home = self.nccl(pick)
        if not nccl_home:
            return None
        cuda_home = self.nvcc(pick_cuda(pick) or cuda)
        if not cuda_home:
            return None
        if os.path.isdir(tests_dir):
            shutil.rmtree(tests_dir)
        os.makedirs(tests_dir)
        git_env = {"GIT_TERMINAL_PROMPT": "0", **self.ctx.env_extra()}
        steps = [
            ("install-nccl-tests-git-init", ["git", "init", "-q", tests_dir]),
            ("install-nccl-tests-git-fetch", ["git", "-C", tests_dir, "fetch", "-q", "--depth", "1", lock["repo"],
                                              lock["commit"]]),
            ("install-nccl-tests-git-checkout", ["git", "-C", tests_dir, "-c", "advice.detachedHead=false",
                                                 "checkout", "-q", "FETCH_HEAD"]),
        ]
        for name, cmd in steps:
            r = self.ctx.proc.run(name, cmd, timeout=600, env_extra=git_env)
            if not r.ok:
                self.ctx.add("install.nccl_tests", "error", None, lock["commit"], None, f"{name}: {r.why()}")
                return None
        gencode = " ".join(f"-gencode=arch=compute_{c},code=sm_{c}" for c in sorted(compute_caps))
        make = ["make", "-C", tests_dir, f"-j{jobs or os.cpu_count() or 8}", f"CUDA_HOME={cuda_home}",
                f"NCCL_HOME={nccl_home}"]
        if gencode:
            make.append(f"NVCC_GENCODE={gencode}")
        r = self.ctx.proc.run("install-nccl-tests-make", make, timeout=3600, env_extra=self.ctx.env_extra())
        if not r.ok or not os.path.exists(os.path.join(tests_dir, "build", "all_reduce_perf")):
            self.ctx.add("install.nccl_tests", "error", None, lock["commit"], None, f"build failed: {r.why()}")
            return None
        meta = {"dir": tests_dir, "commit": lock["commit"], "tag": lock["tag"], "nccl_apt_version": pick,
                "nccl_home": nccl_home, "cuda_home": cuda_home}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.done["nccl_tests"] = meta
        self.ctx.add("install.nccl_tests", "pass", None, f"{lock['tag']} ({lock['commit'][:12]})", None,
                     f"built against NCCL {pick}")
        return meta

    def existing_build(self, cuda, nccl_want):
        """A finished build at the pinned commit for this NCCL and CUDA major, so reruns skip apt."""
        major = (cuda or "").split(".")[0]
        prefix = f"nccl-tests-{self.lock['nccl_tests']['commit'][:12]}-nccl-"
        for meta_path in sorted(glob.glob(os.path.join(self.tools_dir, prefix + "*", "gfv-tool.json")), reverse=True):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            v = meta.get("nccl_apt_version", "")
            if nccl_want and not v.startswith(nccl_want + "-"):
                continue
            if f"+cuda{major}." not in v:
                continue
            if os.access(os.path.join(meta["dir"], "build", "all_reduce_perf"), os.X_OK):
                return meta
        return None

    def nccl(self, apt_version):
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", apt_version)
        home = os.path.join(self.tools_dir, f"nccl-{safe}")
        if os.path.exists(os.path.join(home, ".complete")):
            self.done["nccl"] = {"version": apt_version, "home": home}
            self.ctx.add("install.nccl", "info", None, apt_version, None, "already unpacked")
            return home
        if os.path.isdir(home):
            shutil.rmtree(home)
        os.makedirs(home)
        tmp = tempfile.mkdtemp(prefix="gfv-nccl-", dir=self.tools_dir)
        try:
            r = self.ctx.proc.run("install-nccl-download", ["apt-get", "download", f"libnccl2={apt_version}",
                                                            f"libnccl-dev={apt_version}"],
                                  timeout=900, env_extra={**APT_ENV, **self.ctx.env_extra()}, cwd=tmp)
            debs = sorted(glob.glob(os.path.join(tmp, "*.deb")))
            if not r.ok or len(debs) < 2:
                self.ctx.add("install.nccl", "error", None, apt_version, None, f"apt-get download: {r.why()}")
                return None
            for deb in debs:
                x = self.ctx.proc.run("install-nccl-extract", ["dpkg-deb", "-x", deb, home], timeout=300)
                if not x.ok:
                    self.ctx.add("install.nccl", "error", None, apt_version, None, f"dpkg-deb: {x.why()}")
                    return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        libs = glob.glob(os.path.join(home, "usr", "lib", "**", "libnccl.so*"), recursive=True)
        hdrs = glob.glob(os.path.join(home, "usr", "include", "**", "nccl.h"), recursive=True)
        if not libs or not hdrs:
            self.ctx.add("install.nccl", "error", None, apt_version, None, "libnccl.so or nccl.h missing after extract")
            return None
        os.symlink(os.path.relpath(os.path.dirname(libs[0]), home), os.path.join(home, "lib"))
        os.symlink(os.path.relpath(os.path.dirname(hdrs[0]), home), os.path.join(home, "include"))
        open(os.path.join(home, ".complete"), "w").close()
        self.done["nccl"] = {"version": apt_version, "home": home}
        self.ctx.add("install.nccl", "pass", None, apt_version, None, f"unpacked into {home} (system NCCL untouched)")
        return home

    def nvcc(self, cuda):
        """A CUDA toolkit with the same major version as the NCCL build, else install nvcc for it."""
        candidates = [f"/usr/local/cuda-{cuda}", "/usr/local/cuda"]
        found = self.ctx.which("nvcc")
        if found:
            candidates.append(os.path.dirname(os.path.dirname(found)))
        for home in candidates:
            if self.ctx.sys.exists(f"{home}/bin/nvcc") and toolkit_major(self.ctx.sys, home) == cuda.split(".")[0]:
                self.done["nvcc"] = self.ctx.sys.path(home)
                self.ctx.add("install.nvcc", "info", None, home, None, "present")
                return self.ctx.sys.path(home)
        dashed = cuda.replace(".", "-")
        pkgs = [f"cuda-nvcc-{dashed}", f"cuda-cudart-dev-{dashed}"]
        r = self._install("install-nvcc", pkgs)
        home = f"/usr/local/cuda-{cuda}"
        if r.ok and self.ctx.sys.exists(f"{home}/bin/nvcc"):
            self.done["apt_packages"] += pkgs
            self.done["nvcc"] = self.ctx.sys.path(home)
            self.ctx.add("install.nvcc", "pass", None, pkgs, None, "installed")
            return self.ctx.sys.path(home)
        self.ctx.add("install.nvcc", "error", None, pkgs, None, r.why() if not r.ok else f"{home}/bin/nvcc missing")
        return None

    def write_record(self):
        with open(os.path.join(self.ctx.out_dir, "installed.json"), "w") as f:
            json.dump(self.done, f, indent=2)


def pick_nccl(versions, want, cuda):
    """Pick a libnccl2 apt version: requested NCCL (or newest) built for the driver's CUDA major, with the
    highest CUDA minor that does not exceed the driver's."""
    try:
        major, minor = (int(x) for x in cuda.split(".")[:2])
    except (AttributeError, ValueError):
        return None
    best = None
    for v in versions:
        m = re.match(r"^(\d+\.\d+\.\d+)-\d+\+cuda(\d+)\.(\d+)$", v)
        if not m or int(m.group(2)) != major or int(m.group(3)) > minor:
            continue
        if want and m.group(1) != want:
            continue
        key = (int(m.group(3)), tuple(int(x) for x in m.group(1).split(".")))
        if best is None or key > best[0]:
            best = (key, v)
    return best[1] if best else None


def toolkit_major(sysfs, home):
    try:
        doc = json.loads(sysfs.read(f"{home}/version.json") or "")
        return str(doc["cuda"]["version"]).split(".")[0]
    except (ValueError, KeyError, TypeError):
        return None


def apt_simulation_problems(text):
    """`apt-get -s install` output -> (reasons to refuse, other upgrades, new packages).
    Refused: any removal; a new driver/kernel/CUDA-driver/NCCL/OFED package; an upgrade of the stack under
    test. Other upgrades of installed packages are allowed and returned as "pkg old -> new"."""
    problems, upgrades, new = [], [], []
    for line in (text or "").splitlines():
        m = re.match(r"^(Inst|Remv|Purg)\s+(\S+)(?:\s+\[([^\]]+)\])?(?:\s+\((\S+))?", line)
        if not m:
            continue
        op, pkg, old, newver = m.groups()
        if op in ("Remv", "Purg"):
            problems.append(f"would remove {pkg}")
        elif old and PROTECTED_UPGRADE.match(pkg):
            problems.append(f"would upgrade {pkg} ({old} -> {newver}), part of the stack under test")
        elif old:
            upgrades.append(f"{pkg} {old} -> {newver}")
        elif PROTECTED_NEW.match(pkg):
            problems.append(f"would install protected package {pkg}")
        else:
            new.append(pkg)
    return problems, upgrades, new


def pick_cuda(apt_version):
    m = re.search(r"\+cuda(\d+\.\d+)$", apt_version or "")
    return m.group(1) if m else None


def run(ctx, tools_dir, nccl_needed=True, fetch=_fetch, machine=None):
    """Install stage. Sets ctx.nccl_tests / ctx.ld_path when nccl-tests is available."""
    ctx.stage("install", f"test tools into {tools_dir}")
    os.makedirs(tools_dir, exist_ok=True)
    inst = Installer(ctx, tools_dir, fetch=fetch, machine=machine)
    try:
        if not inst.supported():
            return inst
        cuda = ctx.facts.get("cuda") or ""
        inst.ensure_ca()
        # Add the CUDA repo before the first apt-get update when anything will come from it.
        if not ctx.which("dcgmi") or (nccl_needed and not inst.existing_build(cuda, ctx.spec.get("nccl"))):
            inst.ensure_cuda_repo()
        try:
            inst.apt_tools()
            inst.dcgm(cuda.split(".")[0])
            if nccl_needed:
                caps = {g["compute_cap"].replace(".", "") for g in ctx.facts.get("gpus", [])
                        if re.match(r"^\d+\.\d+$", g.get("compute_cap", ""))}
                meta = inst.nccl_tests(cuda, ctx.spec.get("nccl"), caps)
                if meta:
                    use_tools(ctx, meta)
        finally:
            inst.remove_cuda_repo()
        return inst
    finally:
        inst.write_record()


def use_tools(ctx, meta):
    ctx.nccl_tests = meta["dir"]
    ctx.ld_path = [os.path.join(meta["nccl_home"], "lib")]
    ctx.rec.versions["nccl_tests"] = f"{meta.get('tag')} ({meta.get('commit', '')[:12]})"
