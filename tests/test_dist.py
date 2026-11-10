"""The single-file build: bundled manifests, stdin execution (curl | bash), args, temp cleanup."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from helpers import REPO, FakeNode
from test_accept import REFERENCE_FAILS

try:
    import tomllib  # noqa: F401  (the build converts TOML manifests; Python 3.10 cannot)
    HAVE_TOMLLIB = True
except ImportError:
    HAVE_TOMLLIB = False


@unittest.skipUnless(HAVE_TOMLLIB, "building the single file needs Python 3.11+; runtime tests cover 3.10")
class TestDist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gfv-dist-")
        env = dict(os.environ, DIST_DIR=cls.tmp)
        subprocess.run(["bash", os.path.join(REPO, "scripts", "build-dist.sh")], check=True, env=env,
                       capture_output=True)
        cls.dist = os.path.join(cls.tmp, "gfv-accept.sh")

    def setUp(self):
        self.node = FakeNode().build()
        self.addCleanup(self.node.cleanup)
        self.tmpdir = os.path.join(self.node.tmp, "wrapper-tmp")
        os.makedirs(self.tmpdir)

    def env(self):
        pybin = os.path.join(self.node.tmp, "py")
        os.makedirs(pybin, exist_ok=True)
        link = os.path.join(pybin, "python3")
        if not os.path.exists(link):
            os.symlink(sys.executable, link)
        env = self.node.env(TMPDIR=self.tmpdir)
        env["PATH"] = os.pathsep.join([self.node.bin, pybin, "/usr/bin", "/bin"])
        del env["PYTHONPATH"]           # the payload must be self-contained
        return env

    def args(self):
        return ["accept", "--class", "hgx-h100-8x", "--no-install", "--node", "gpu-node-01",
                "--out", self.node.out, "--tools-dir", self.node.tools_dir]

    def results(self):
        with open(os.path.join(self.node.out, "results.jsonl")) as f:
            return [json.loads(line) for line in f]

    def failing(self):
        return sorted((r["check"], r["target"]) for r in self.results() if r["verdict"] in ("fail", "error"))

    def test_file_run_uses_bundled_manifest(self):
        p = subprocess.run(["bash", self.dist, *self.args()], env=self.env(), capture_output=True, text=True,
                           timeout=120, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.failing(), REFERENCE_FAILS)
        self.assertIn("bundled:hgx-h100-8x.json", p.stderr)
        self.assertEqual(os.listdir(self.tmpdir), [], "wrapper must remove its temp dir")

    def test_stdin_run_like_curl_pipe(self):
        with open(self.dist) as f:
            script = f.read()
        p = subprocess.run(["bash", "-s", "--", *self.args()], input=script, env=self.env(), capture_output=True,
                           text=True, timeout=120, cwd=self.node.tmp)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.failing(), REFERENCE_FAILS)
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_truncated_download_runs_nothing(self):
        with open(self.dist) as f:
            script = f.read()
        cut = script[: len(script) // 2]
        p = subprocess.run(["bash", "-s", "--", *self.args()], input=cut, env=self.env(), capture_output=True,
                           text=True, timeout=60, cwd=self.node.tmp)
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(self.node.out, "results.jsonl")))
        self.assertEqual(self.node.calls(), [])

    def test_version_and_exit_codes_pass_through(self):
        p = subprocess.run(["bash", self.dist, "accept", "--class", "nope"], env=self.env(), capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(p.returncode, 2)
        p = subprocess.run(["bash", self.dist, "version"], env=self.env(), capture_output=True, text=True)
        self.assertRegex(p.stdout, r"^gfv \d+\.\d+\.\d+ \(")


if __name__ == "__main__":
    unittest.main()
