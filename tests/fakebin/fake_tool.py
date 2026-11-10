"""Stand-in for every external tool the runner calls. Serves recorded fixtures; never runs a real tool.

Environment:
  GFV_FAKE_DIR    scenario directory (a copy of tests/fixtures/h100-8x-ib, possibly edited by a test)
  GFV_FAKE_STATE  writable directory: calls.log (one JSON array per call) and markers
  GFV_SYSROOT     fake root; `dcgmi diag` applies <scenario>/after-load.json to it (counters after load)
"""

import json
import os
import sys

TOOL, ARGS = sys.argv[1], sys.argv[2:]
SCEN = os.environ["GFV_FAKE_DIR"]
STATE = os.environ["GFV_FAKE_STATE"]
ROOT = os.environ.get("GFV_SYSROOT", "/nonexistent")

with open(os.path.join(STATE, "calls.log"), "a") as f:
    f.write(json.dumps([TOOL, *ARGS, {"cwd": os.getcwd()}]) + "\n")


def fixture(name):
    with open(os.path.join(SCEN, name)) as f:
        return f.read()


def has(name):
    return os.path.exists(os.path.join(SCEN, name))


def marker(name):
    return os.path.exists(os.path.join(STATE, name))


def touch(name):
    open(os.path.join(STATE, name), "w").close()


def out(text, rc=0):
    sys.stdout.write(text)
    sys.exit(rc)


def phase(name, ext):
    """<name>-after.<ext> once the load ran (if the scenario has one), else <name>.<ext>."""
    after = f"{name}-after.{ext}"
    return fixture(after) if marker("load-done") and has(after) else fixture(f"{name}.{ext}")


def after_load():
    touch("load-done")
    if has("after-load.json"):
        for path, content in json.loads(fixture("after-load.json")).items():
            p = os.path.join(ROOT, path.lstrip("/"))
            with open(p, "w") as f:
                f.write(content)


def nvidia_smi():
    if not ARGS:
        out(fixture("nvidia-smi.txt"))
    a = ARGS[0]
    if a.startswith("--query-gpu="):
        want = fixture("gpus.fields").strip()
        if a.split("=", 1)[1] != want:
            out(f'Field "{a}" is not a valid field to query.\n', 2)
        out(phase("gpus", "csv"))
    if a.startswith("--query-remapped-rows="):
        out(phase("remapped", "csv"))
    if a.startswith("--query-compute-apps="):
        out(fixture("compute-apps.csv"))
    if ARGS == ["topo", "-m"]:
        out(fixture("topo.txt"))
    if ARGS == ["nvlink", "-s"]:
        out(fixture("nvlink-s.txt"))
    if ARGS == ["nvlink", "-e"]:
        out(fixture("nvlink-e-after.txt") if marker("load-done") else fixture("nvlink-e.txt"))
    if ARGS == ["-q"]:
        out(fixture("smi-q.txt"))
    out(f"fake nvidia-smi: unhandled {ARGS}\n", 9)


def dcgmi():
    if ARGS == ["--version"]:
        out(fixture("dcgmi-version.txt"))
    if has("no-hostengine") and not marker("hostengine"):
        out("Error: unable to establish a connection to the specified host: localhost\n", 255)
    if ARGS[:2] == ["discovery", "-l"]:
        out(fixture("dcgmi-discovery.txt"))
    if ARGS[:1] == ["diag"]:
        after_load()
        out(fixture("dcgm-diag-r3.json"), 226 if has("dcgm-rc-fail") else 0)
    out(f"fake dcgmi: unhandled {ARGS}\n", 9)


def ib_write_bw():
    if "--help" in ARGS:
        out(fixture("ib-write-bw-help.txt"))
    if ARGS[-1] != "127.0.0.1":
        out("server done\n")                      # server side: the client prints the numbers
    dev = ARGS[ARGS.index("-d") + 1]
    bw = json.loads(fixture("rdma-bw.json"))
    if dev in bw.get("unreachable", []):
        out("Couldn't connect to 127.0.0.1:18515\nUnable to init the socket connection\n", 1)
    avg = bw["per_client"].get(dev, bw["default"])
    out(" #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n"
        f" 65536      2090000          0.00               {avg:.2f} \t\t     0.697\n")


def simple():
    table = {
        "mokutil": ("mokutil.txt", 0),
        "ofed_info": ("ofed-info.txt", 0),
        "dpkg-query": ("fabricmanager-dpkg.txt", 0),
    }
    name, rc = table[TOOL]
    out(fixture(name), rc)


def main():
    if TOOL == "nvidia-smi":
        nvidia_smi()
    elif TOOL == "dcgmi":
        dcgmi()
    elif TOOL == "nv-hostengine":
        if "--term" in ARGS:
            touch("hostengine-stopped")
        else:
            touch("hostengine")
        out("")
    elif TOOL == "ib_write_bw":
        ib_write_bw()
    elif TOOL == "all_reduce_perf":
        touch("load-done")
        out(fixture("all_reduce_perf.log"))
    elif TOOL == "lspci":
        out(fixture("lspci-vvv.txt") if "-vvv" in ARGS else fixture("lspci-nvidia.txt"))
    elif TOOL == "systemctl":
        text = fixture("systemctl-fabricmanager.txt")
        out(text, 0 if text.strip() == "active" else 3)
    elif TOOL == "dmesg":
        out(fixture("dmesg-after.txt") if marker("load-done") else fixture("dmesg-before.txt"))
    elif TOOL == "fio":
        rw = next(a.split("=", 1)[1] for a in ARGS if a.startswith("--rw="))
        out(fixture(f"fio-{rw}.json"))
    elif TOOL in ("mokutil", "ofed_info", "dpkg-query"):
        simple()
    else:
        install_tools()


# --- package-management fakes (install tests) -------------------------------------------------
def install_tools():
    if TOOL == "apt-get":
        if ARGS[:1] == ["download"]:
            for spec in ARGS[1:]:
                name, ver = spec.split("=")
                open(f"{name}_{ver.replace(':', '%3a')}_amd64.deb", "w").close()
            out("")
        if ARGS[:1] == ["install"]:
            for pkg in ARGS:
                if pkg.startswith("cuda-nvcc-"):
                    cuda = pkg[len("cuda-nvcc-"):].replace("-", ".")
                    d = os.path.join(ROOT, f"usr/local/cuda-{cuda}/bin")
                    os.makedirs(d, exist_ok=True)
                    open(os.path.join(d, "nvcc"), "w").close()
            if has("apt-install-fail") and any(p.startswith(fixture("apt-install-fail").strip()) for p in ARGS):
                out("E: Version not found\n", 100)
        out("")
    if TOOL == "apt-cache":
        out(fixture("apt-madison.txt"))
    if TOOL == "dpkg":
        d = os.path.join(ROOT, "etc/apt/sources.list.d")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "cuda-ubuntu2404-x86_64.list"), "w") as f:
            f.write("deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] "
                    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /\n")
        out("")
    if TOOL == "dpkg-deb":
        deb, dest = ARGS[1], ARGS[2]
        if os.path.basename(deb).startswith("libnccl-dev"):
            os.makedirs(os.path.join(dest, "usr/include"), exist_ok=True)
            open(os.path.join(dest, "usr/include/nccl.h"), "w").close()
            libdir = os.path.join(dest, "usr/lib/x86_64-linux-gnu")
            os.makedirs(libdir, exist_ok=True)
            open(os.path.join(libdir, "libnccl.so"), "w").close()
        else:
            libdir = os.path.join(dest, "usr/lib/x86_64-linux-gnu")
            os.makedirs(libdir, exist_ok=True)
            open(os.path.join(libdir, "libnccl.so.2"), "w").close()
        out("")
    if TOOL == "git":
        if ARGS[:2] == ["init", "-q"]:
            os.makedirs(ARGS[2], exist_ok=True)
        out("")
    if TOOL == "make":
        d = ARGS[ARGS.index("-C") + 1]
        os.makedirs(os.path.join(d, "build"), exist_ok=True)
        exe = os.path.join(d, "build", "all_reduce_perf")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(exe, 0o755)
        out("")
    out(f"fake: no handler for {TOOL} {ARGS}\n", 9)


main()
