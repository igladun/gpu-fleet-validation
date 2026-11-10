"""The release build: gfv.pyz, the gfv.sh launcher (hash check, curl | bash, args, temp cleanup), reproducibility."""

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import zipfile

from helpers import REPO, FakeNode
from test_accept import REFERENCE_FAILS

from gfv import __version__

try:
    import tomllib  # noqa: F401  (the build converts TOML manifests; Python 3.10 cannot)
    HAVE_TOMLLIB = True
except ImportError:
    HAVE_TOMLLIB = False

ASSETS = ("gfv.pyz", "gfv.sh", "SHA256SUMS")
RELEASE_URL = f"https://github.com/igladun/gpu-fleet-validation/releases/download/v{__version__}/gfv.pyz"
UNSET = object()

# Stands in for curl and wget: logs its argv, then serves FAKE_DL_SRC, fails, or serves a captive-portal page.
FAKE_DOWNLOADER = """#!/usr/bin/env python3
import json, os, shutil, sys
name, args = os.path.basename(sys.argv[0]), sys.argv[1:]
with open(os.environ["FAKE_DL_LOG"], "a") as f:
    f.write(json.dumps([name, *args]) + "\\n")
dest = args[args.index("-o" if name == "curl" else "-O") + 1]
mode = os.environ.get("FAKE_DL_MODE", "ok")
if mode == "fail":
    sys.exit(22 if name == "curl" else 8)
if mode == "html":
    with open(dest, "w") as f:
        f.write("<html>sign in to the hotel wifi</html>")
    sys.exit(0)
shutil.copyfile(os.environ["FAKE_DL_SRC"], dest)
"""

# A toy zipapp: prints its pid, then on SIGTERM imports a module from its own archive and exits 42.
TOY_MAIN = b"""import os, signal, sys, time
def on_term(sig, frame):
    import toy_late
    sys.exit(toy_late.CODE)
signal.signal(signal.SIGTERM, on_term)
print(os.getpid(), flush=True)
time.sleep(60)
sys.exit(1)
"""


def build(out_dir):
    env = dict(os.environ, DIST_DIR=out_dir)
    subprocess.run(["bash", os.path.join(REPO, "scripts", "build-dist.sh")], check=True, env=env,
                   capture_output=True)


@unittest.skipUnless(HAVE_TOMLLIB, "building the release needs Python 3.11+; runtime tests cover 3.10")
class TestDist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gfv-dist-")
        build(cls.tmp)
        cls.pyz = os.path.join(cls.tmp, "gfv.pyz")
        cls.stub = os.path.join(cls.tmp, "gfv.sh")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.node = FakeNode().build()
        self.addCleanup(self.node.cleanup)
        self.tmpdir = os.path.join(self.node.tmp, "wrapper-tmp")
        os.makedirs(self.tmpdir)

    def pybin(self):
        pybin = os.path.join(self.node.tmp, "py")
        os.makedirs(pybin, exist_ok=True)
        link = os.path.join(pybin, "python3")
        if not os.path.exists(link):
            os.symlink(sys.executable, link)
        return pybin

    def env(self, pyz=None, path=None):
        """pyz: GFV_PYZ (default: the built gfv.pyz; UNSET: leave it unset, so the launcher downloads)."""
        env = self.node.env(TMPDIR=self.tmpdir)
        if pyz is not UNSET:
            env["GFV_PYZ"] = pyz if pyz is not None else self.pyz
        env["PATH"] = os.pathsep.join(path or [self.node.bin, self.pybin(), "/usr/bin", "/bin"])
        del env["PYTHONPATH"]           # the payload must be self-contained
        return env

    def downloader(self, *names, mode="ok"):
        """A directory holding fake curl and/or wget, and the env vars that drive them."""
        d = os.path.join(self.node.tmp, "dl-" + "-".join(names))
        os.makedirs(d)
        for name in names:
            with open(os.path.join(d, name), "w") as f:
                f.write(FAKE_DOWNLOADER)
            os.chmod(os.path.join(d, name), 0o755)
        log = os.path.join(self.node.tmp, "dl.log")
        return d, {"FAKE_DL_LOG": log, "FAKE_DL_SRC": self.pyz, "FAKE_DL_MODE": mode}, log

    def args(self):
        return ["accept", "--class", "hgx-h100-8x", "--no-install", "--node", "gpu-node-01",
                "--out", self.node.out, "--tools-dir", self.node.tools_dir]

    def results(self):
        with open(os.path.join(self.node.out, "results.jsonl")) as f:
            return [json.loads(line) for line in f]

    def failing(self):
        return sorted((r["check"], r["target"]) for r in self.results() if r["verdict"] in ("fail", "error"))

    def stub_text(self):
        with open(self.stub) as f:
            return f.read()

    def test_file_run_uses_bundled_manifest(self):
        p = subprocess.run(["bash", self.stub, *self.args()], env=self.env(), capture_output=True, text=True,
                           timeout=120, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.failing(), REFERENCE_FAILS)
        self.assertIn("bundled:hgx-h100-8x.json", p.stderr)
        self.assertIn("verified", p.stderr)
        self.assertEqual(os.listdir(self.tmpdir), [], "launcher must remove its temp dir")

    def test_stdin_run_like_curl_pipe(self):
        p = subprocess.run(["bash", "-s", "--", *self.args()], input=self.stub_text(), env=self.env(),
                           capture_output=True, text=True, timeout=120, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.failing(), REFERENCE_FAILS)
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_pyz_runs_directly(self):
        """The offline path: copy gfv.pyz to the node and run it with python3."""
        p = subprocess.run([sys.executable, "-I", self.pyz, *self.args()], env=self.env(), capture_output=True,
                           text=True, timeout=120, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.failing(), REFERENCE_FAILS)

    def test_hash_mismatch_refuses(self):
        tampered = os.path.join(self.node.tmp, "gfv.pyz")
        with open(self.pyz, "rb") as f:
            data = f.read()
        with open(tampered, "wb") as f:
            f.write(data + b"\0")
        p = subprocess.run(["bash", "-s", "--", *self.args()], input=self.stub_text(), env=self.env(tampered),
                           capture_output=True, text=True, timeout=60, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("sha256 mismatch", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.node.out, "results.jsonl")))
        self.assertEqual(self.node.calls(), [])
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_missing_local_copy_refuses(self):
        p = subprocess.run(["bash", self.stub, "version"], env=self.env(os.path.join(self.node.tmp, "nope.pyz")),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(p.stdout, "")
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_truncated_download_runs_nothing(self):
        """The body is one { ... } block: any cut after the shebang line is a syntax error (exit 2)."""
        script = self.stub_text()
        first = script.index("\n") + 1
        lines = [i + 1 for i, c in enumerate(script) if c == "\n"]
        cuts = sorted({c for c in lines + [i + 7 for i in lines] if first < c < len(script) - 1})
        self.assertGreater(len(cuts), 100)
        for cut in cuts:
            p = subprocess.run(["bash", "-s", "--", *self.args()], input=script[:cut], env=self.env(),
                               capture_output=True, text=True, timeout=60, cwd=self.node.tmp)
            self.assertEqual(p.returncode, 2, f"cut at {cut}: {script[max(0, cut - 40):cut]!r}\n{p.stderr}")
            self.assertEqual(p.stdout, "", f"cut at {cut}")
        self.assertFalse(os.path.exists(os.path.join(self.node.out, "results.jsonl")))
        self.assertEqual(self.node.calls(), [])
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_empty_local_copy_refuses(self):
        p = subprocess.run(["bash", self.stub, "version"], env=self.env(""), capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("GFV_PYZ is set but empty", p.stderr)

    def run_download(self, path, extra, *args):
        env = self.env(UNSET, path=path)
        env.update(extra)
        return subprocess.run([shutil.which("bash"), self.stub, *args], env=env, capture_output=True, text=True, timeout=60,
                              cwd=self.node.tmp)

    def test_download_with_curl(self):
        d, extra, log = self.downloader("curl")
        p = self.run_download([d, self.node.bin, self.pybin(), "/usr/bin", "/bin"], extra, "version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout, rf"^gfv {__version__} ")
        with open(log) as f:
            (argv,) = [json.loads(line) for line in f]
        self.assertEqual(argv[0], "curl")
        self.assertEqual(argv[-1], RELEASE_URL)
        self.assertEqual(argv[argv.index("--proto") + 1], "=https")
        self.assertIn("-f", " ".join(argv[1:]).replace("-fsSL", "-f -s -S -L").split())
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_download_failure_refuses(self):
        for mode, why in (("fail", "download failed"), ("html", "sha256 mismatch")):
            with self.subTest(mode=mode):
                d, extra, _ = self.downloader("curl", mode=mode)
                p = self.run_download([d, self.node.bin, self.pybin(), "/usr/bin", "/bin"], extra, *self.args())
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn(why, p.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.node.out, "results.jsonl")))
                self.assertEqual(self.node.calls(), [])
                self.assertEqual(os.listdir(self.tmpdir), [])
                shutil.rmtree(d)

    def test_download_with_wget_when_no_curl(self):
        tools = os.path.join(self.node.tmp, "coreutils")
        os.makedirs(tools)
        for name in ("mktemp", "cp", "rm"):
            os.symlink(shutil.which(name), os.path.join(tools, name))
        d, extra, log = self.downloader("wget")
        p = self.run_download([d, self.pybin(), tools], extra, "version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout, rf"^gfv {__version__} ")
        with open(log) as f:
            (argv,) = [json.loads(line) for line in f]
        self.assertEqual((argv[0], argv[-1]), ("wget", RELEASE_URL))

    def test_launcher_execs_python(self):
        """The launcher replaces itself with python3: same pid, signals reach the runner, and the runner can
        still import from its archive after the launcher deleted the file."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("__main__.py", TOY_MAIN)
            z.writestr("toy_late.py", b"CODE = 42\n")
        toy = os.path.join(self.node.tmp, "toy.pyz")
        with open(toy, "wb") as f:
            f.write(buf.getvalue())
        with open(os.path.join(REPO, "scripts", "bootstrap.sh.in")) as f:
            text = f.read().replace("@VERSION@", "9.9.9").replace("@SHA256@", hashlib.sha256(buf.getvalue()).hexdigest())
        stub = os.path.join(self.node.tmp, "toy.sh")
        with open(stub, "w") as f:
            f.write(text)
        proc = subprocess.Popen(["bash", stub], env=self.env(toy), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        try:
            pid = int(proc.stdout.readline())
            self.assertEqual(pid, proc.pid, "launcher must exec python3, not run it as a child")
            self.assertEqual(os.listdir(self.tmpdir), [], "the archive must be deleted before python3 runs")
            proc.send_signal(signal.SIGTERM)
            self.assertEqual(proc.wait(timeout=20), 42, proc.stderr.read())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()

    def test_stub_pins_release(self):
        text = self.stub_text()
        with open(self.pyz, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertIn(f'GFV_VERSION="{__version__}"', text)
        self.assertIn(f'GFV_SHA256="{sha}"', text)
        self.assertIn(f"/releases/download/v{__version__}/gfv.pyz", text)
        self.assertNotRegex(text, r"@[A-Z0-9_]+@")
        with open(os.path.join(self.tmp, "SHA256SUMS")) as f:
            sums = dict(reversed(line.split()) for line in f)
        self.assertEqual(sums["gfv.pyz"], sha)
        self.assertEqual(sums["gfv.sh"], hashlib.sha256(text.encode()).hexdigest())

    def test_reproducible_build(self):
        other = tempfile.mkdtemp(prefix="gfv-dist2-")
        self.addCleanup(shutil.rmtree, other, True)
        build(other)
        for name in ASSETS:
            with self.subTest(asset=name):
                with open(os.path.join(self.tmp, name), "rb") as a, open(os.path.join(other, name), "rb") as b:
                    self.assertTrue(a.read() == b.read(), f"{name} differs between two builds")

    def test_version_and_exit_codes_pass_through(self):
        p = subprocess.run(["bash", self.stub, "accept", "--class", "nope"], env=self.env(), capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(p.returncode, 2)
        p = subprocess.run(["bash", self.stub, "version"], env=self.env(), capture_output=True, text=True)
        self.assertRegex(p.stdout, rf"^gfv {__version__} \(src [0-9a-f]{{12}}\)")


if __name__ == "__main__":
    unittest.main()
