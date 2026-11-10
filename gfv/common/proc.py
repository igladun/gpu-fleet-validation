"""Run external tools and keep every output as evidence."""

import os
import re
import shlex
import subprocess
import time

from .results import utcnow


class CmdResult:
    def __init__(self, name, cmd, rc, out, err, seconds, missing=False, timed_out=False):
        self.name, self.cmd, self.rc, self.out, self.err = name, cmd, rc, out, err
        self.seconds, self.missing, self.timed_out = seconds, missing, timed_out
        self.evidence = f"raw/{name}.out"

    @property
    def ok(self):
        return self.rc == 0

    def why(self):
        """Short reason for a failed command, for result details."""
        if self.missing:
            return f"{self.cmd[0]} not found"
        if self.timed_out:
            return f"timed out after {self.seconds:.0f} s"
        tail = (self.err or self.out or "").strip().splitlines()[-1:] or [""]
        return f"rc={self.rc} {tail[0][:200]}".strip()


class Proc:
    """Runs commands with stdin closed, a timeout, and the output saved under <evidence>/raw/."""

    def __init__(self, evidence_dir, env=None):
        self.raw = os.path.join(evidence_dir, "raw")
        os.makedirs(self.raw, exist_ok=True)
        self.env = dict(os.environ if env is None else env)
        self._log = os.path.join(self.raw, "commands.log")
        self._seen = {}

    def _name(self, name):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        n = self._seen.get(name, 0)
        self._seen[name] = n + 1
        return name if n == 0 else f"{name}.{n}"

    def run(self, name, cmd, timeout=120, env_extra=None, cwd=None, to_file=False):
        """Run cmd. to_file=True sends output straight to the evidence file instead of a pipe, for
        programs that daemonize and keep their stdout open (a pipe read would never finish)."""
        name = self._name(name)
        env = dict(self.env, **(env_extra or {}))
        start = time.monotonic()
        missing = timed_out = False
        out_path = os.path.join(self.raw, f"{name}.out")
        try:
            if to_file:
                with open(out_path, "w") as fh:
                    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=timeout, env=env, cwd=cwd, start_new_session=True)
                with open(out_path, errors="replace") as fh:
                    rc, out, err = p.returncode, fh.read(), ""
            else:
                p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                   errors="replace", timeout=timeout, env=env, cwd=cwd)
                rc, out, err = p.returncode, p.stdout, p.stderr
        except FileNotFoundError as e:
            rc, out, err, missing = 127, "", str(e), True
        except subprocess.TimeoutExpired as e:
            rc, timed_out = 124, True
            out = _text(e.stdout)
            if to_file:
                with open(out_path, errors="replace") as fh:
                    out = fh.read()
            err = _text(e.stderr) + f"\n[gfv] timed out after {timeout} s"
        except OSError as e:          # not executable, permission denied, ...
            rc, out, err = 126, "", f"{type(e).__name__}: {e}"
        secs = time.monotonic() - start
        self._save(name, cmd, rc, out, err, secs)
        return CmdResult(name, cmd, rc, out, err, secs, missing, timed_out)

    def spawn(self, name, cmd, env_extra=None):
        """Start a background process (e.g. a perftest server); stop it with finish()."""
        name = self._name(name)
        env = dict(self.env, **(env_extra or {}))
        out = open(os.path.join(self.raw, f"{name}.out"), "w")
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env)
        except OSError:
            out.close()
            return None
        return _Background(self, name, cmd, p, out)

    def _save(self, name, cmd, rc, out, err, secs):
        with open(os.path.join(self.raw, f"{name}.out"), "w") as f:
            f.write(out or "")
        if err:
            with open(os.path.join(self.raw, f"{name}.err"), "w") as f:
                f.write(err)
        with open(self._log, "a") as f:
            f.write(f"{utcnow()} rc={rc} {secs:.1f}s {name}: {shlex.join(cmd)}\n")


class _Background:
    def __init__(self, proc, name, cmd, popen, fh):
        self.proc, self.name, self.cmd, self.p, self.fh = proc, name, cmd, popen, fh
        self.start = time.monotonic()

    def finish(self, timeout=15):
        try:
            rc = self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait()
        self.fh.close()
        with open(self.proc._log, "a") as f:
            f.write(f"{utcnow()} rc={rc} {time.monotonic() - self.start:.1f}s {self.name} (background): "
                    f"{shlex.join(self.cmd)}\n")
        return rc


def _text(b):
    if b is None:
        return ""
    return b.decode(errors="replace") if isinstance(b, bytes) else b
