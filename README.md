# gpu-fleet-validation

Scripts that validate multi-node GPU training clusters: 8-GPU HGX-class nodes with one RDMA NIC per GPU, on an InfiniBand or RoCE fabric.

## Run it

Run it on the GPU node itself. It needs Ubuntu 22.04 or 24.04 and a working NVIDIA driver; it installs everything else.

Each [release](https://github.com/igladun/gpu-fleet-validation/releases) has three files:
- `gfv.sh`: a short launcher you can read in a minute. It downloads `gfv.pyz` from the same release, checks its sha256 and runs it with `python3`. It installs `python3` first if the node lacks it.
- `gfv.pyz`: the code, a Python zipapp built from [`gfv/`](gfv/) and [`manifests/`](manifests/). List it with `python3 -m zipfile -l gfv.pyz`.
- `SHA256SUMS`.

**See what it would do.** This needs no root, checks nothing and changes nothing:

```sh
curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | bash -s -- accept --class hgx-h100-8x --dry-run
```

**Accept a node.** This runs as root and takes about 30–40 minutes. Stop GPU workloads first; the gate refuses to run while GPU processes exist:

```sh
curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | sudo bash -s -- accept --class hgx-h100-8x
```

**Over ssh**, so the run survives a dropped session:

```sh
curl -fsSLO https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh
sudo nohup bash gfv.sh accept --class hgx-h100-8x > accept.jsonl 2> accept.log &
tail -f accept.log
```

**When the node can't reach GitHub**, copy `gfv.sh` and `gfv.pyz` from the same release to the node. `GFV_PYZ` makes the launcher use the local copy, and it still checks the sha256 it carries:

```sh
sudo GFV_PYZ="$PWD/gfv.pyz" bash gfv.sh accept --class hgx-h100-8x
```

The install stage still needs apt, NVIDIA's CUDA repo and github.com for nccl-tests. Where those are blocked, install the tools yourself and add `--no-install`.

More:
- **Pin a version:** replace `latest/download` with `download/<tag>`, e.g. `download/v0.1.0`. Each `gfv.sh` carries the version and sha256 of its own `gfv.pyz`.
- **Verify a download:** `sha256sum -c --ignore-missing SHA256SUMS`, or `gh attestation verify gfv.pyz -R igladun/gpu-fleet-validation` to check that the release workflow built it from this repo.
- **Bundled node classes:** list them with `bash gfv.sh classes`.
- **Your own hardware:** pass a manifest file on the node with `--manifest my-class.json` instead of `--class`. Copy and edit [`manifests/hgx-h100-8x.toml`](manifests/hgx-h100-8x.toml); JSON works on every Python, TOML needs 3.11+.
- **Every option:** `bash gfv.sh accept --help`. The checks are documented in [docs/acceptance.md](docs/acceptance.md).

**Exit codes:**
- `0`: the node passed.
- `1`: a check failed or could not run.
- `2`: the gate itself could not run. Causes: the launcher could not download or verify `gfv.pyz`, not root, a bad manifest, no working driver, busy GPUs without `--force`, or an interrupted run.

**In automation, don't trust the exit code of `curl … | bash`.** If curl fails, bash reads an empty script and exits 0. Download first (`curl -fsSLO …/gfv.sh && sudo bash gfv.sh …`) or run the pipe under `set -o pipefail`. A truncated download can't pass: the launcher is one `{ … }` block, so any cut after its first line is a syntax error and exits 2.

**Outputs:**
- **stdout:** one JSON result line per check.
- **Evidence directory:** `/var/lib/gpu-fleet-validation/runs/<run_id>/`, holding `SUMMARY.md`, `results.jsonl`, `junit.xml` and every raw tool output.

> **Status:** the runner is tested against recorded output from a real 8x H100 HGX node, and its installer in Ubuntu containers. It has not yet run on GPU hardware. See [Status of the L2 gate](#status-of-the-l2-gate).

## Test tiers

The work is split into five test tiers. Each tier answers one question at one point in a node's life.

| Tier | Question | When | Status |
|---|---|---|---|
| L0 pod preflight | Can this training pod use its GPUs, NICs and peers? | start of every training job | planned |
| L1 health | Is any GPU, NVLink or IB link degrading? | continuously | planned |
| **L2 node acceptance** | **Is this node fit to take work?** | **node joins, repair, driver / OFED / firmware change** | **[implemented](docs/acceptance.md)** |
| L3 burn-in | Does new or RMA-returned hardware survive hours of load? | new or repaired hardware | planned |
| L4 cluster and fabric | Do collectives and the fabric hold up across nodes? | nightly per block, before big runs | planned |

Rules that hold across all tiers:
- **Expected values live in a node-class manifest** ([`manifests/`](manifests/)), never in test code.
- **Every check writes the same JSON result line.**
- **Fail closed:** a check that cannot run counts as a failure.

## L2 node acceptance gate

L2 checks the node against its class manifest, then puts it under load:

- **Inventory and manifest diff:** GPU count and model; driver and CUDA; ECC; PCIe width; NUMA layout; kernel cmdline; Secure Boot; PCIe ACS; GPUDirect module; firmware.
- **NVLink:** link count and NVSwitch fabric state; NVSwitch count; Fabric Manager status and version.
- **RDMA:** port state and rate; the GPU-to-NIC rail map; `ib_write_bw` loopback through the switch on every rail.
- **DCGM:** `dcgmi diag -r 3`, plus per-GPU PCIe bandwidth floors.
- **NCCL:** single-node `all_reduce_perf`, checking bus bandwidth, correctness and NCCL version.
- **fio:** optional local-storage floors.
- **Health deltas:** XIDs, SXids, ECC, remapped rows, and IB and NVLink error counters, compared before and after the load.

**Tools it installs:** DCGM, perftest, fio and a pinned nccl-tests build. Reruns reuse them.

**What it never touches:** the GPU driver, Fabric Manager, OFED/DOCA, the kernel or the system NCCL.
- Every apt change is simulated first. It is refused if it would remove anything, pull in a driver or kernel package, or upgrade the stack under test.
- Other dependency upgrades are allowed and listed in the results.
- A temporary NVIDIA CUDA apt repo is removed again afterwards.

**Pre-staging:** to install the tools without running the checks, e.g. in an image bake, run with `--only install --allow-partial`.

**Timing:** most of the 30–40 minutes is DCGM level 3. The first run adds about 5–10 minutes for installs.

## Status of the L2 gate

The parsers and the whole runner are tested against recorded output from a real 8x H100 HGX node with NDR InfiniBand. The tests use fake tools and a fake `/sys` tree; see [tests/fixtures/README.md](tests/fixtures/README.md). The installer is tested end to end in Ubuntu containers. The runner has not yet run on GPU hardware. [docs/acceptance.md](docs/acceptance.md#first-hardware-run) lists what that first run must confirm.

**Known finding:** on the reference node, DCGM measured 76 GB/s bidirectional PCIe bandwidth on GPU 0 and GPU 1, against 101 GB/s on GPUs 2–7. The shipped floor is 95 GB/s, so replaying that node fails exactly those two checks. The tests assert that result.

## Development

```sh
python3 -m unittest discover -s tests -v     # Python 3.10+; nothing to install
scripts/build-dist.sh                        # writes dist/gfv.pyz, dist/gfv.sh, dist/SHA256SUMS; needs Python 3.11+
scripts/check-dist.sh dist                   # smoke-tests them with every python3 >= 3.10 on PATH
GFV_PYZ=dist/gfv.pyz bash dist/gfv.sh accept --class hgx-h100-8x --dry-run
python3 -m gfv accept --class hgx-h100-8x --dry-run   # straight from the checkout
```

`gfv.sh` is rendered from [`scripts/bootstrap.sh.in`](scripts/bootstrap.sh.in). `GFV_PYZ` makes it use a local `gfv.pyz` instead of downloading one; the sha256 check still applies. The build is reproducible: the same sources give the same bytes on any OS. Nothing built is committed.

**Releasing:** set `__version__` in [`gfv/__init__.py`](gfv/__init__.py), commit to `main`, then push a tag `v<version>`. The [release workflow](.github/workflows/release.yml) tests and builds in a read-only job. A second job attests the files, uploads them to a draft release, checks the draft against the build, publishes it and runs the published one-liner. A version with a suffix (`v0.2.0-rc1`) becomes a prerelease and never `latest`.

The runtime uses only the Python standard library. The node needs nothing but `python3` ≥ 3.10, and `gfv.sh` installs that on Ubuntu if it is missing.

## License

MIT
