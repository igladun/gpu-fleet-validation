# gpu-fleet-validation

Scripts that validate multi-node GPU training clusters: 8-GPU HGX-class nodes with one RDMA NIC per GPU, on an InfiniBand or RoCE fabric.

## Run it

Run it on the GPU node itself. It needs Ubuntu 22.04, 24.04 or 26.04 and a working NVIDIA driver; it installs everything else.

Each [release](https://github.com/igladun/gpu-fleet-validation/releases) has three files:
- `gfv.sh`: a short launcher you can read in a minute. It downloads `gfv.pyz` from the same release, checks its sha256 and runs it with `python3`. It installs `python3` first if the node lacks it.
- `gfv.pyz`: the code, a Python zipapp built from [`gfv/`](gfv/) and [`manifests/`](manifests/). List it with `python3 -m zipfile -l gfv.pyz`.
- `SHA256SUMS`.

**See what it would do.** This needs no root, checks nothing and changes nothing:

```sh
curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | bash -s -- accept --class hgx-h100-8x --dry-run
```

**Accept a node.** This runs as root and takes about an hour, most of it DCGM level 3 and a 30-minute burn-in at the power limit. Stop GPU workloads first; the gate refuses to run while GPU processes exist:

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
- **Your own hardware:** write a manifest for your node class and pass it with `--manifest my-class.toml` instead of `--class`. See [Add a manifest for your node class](#add-a-manifest-for-your-node-class).
- **Longer burn-in:** `--burnin-minutes 240` holds every GPU at its power limit for 4 hours instead of the manifest's 30 minutes.
- **Every option:** `bash gfv.sh accept --help`. The checks are documented in [docs/acceptance.md](docs/acceptance.md).

**Exit codes:**
- `0`: the node passed.
- `1`: a check failed or could not run.
- `2`: the gate itself could not run. Causes: the launcher could not download or verify `gfv.pyz`, not root, a bad manifest, no working driver, busy GPUs without `--force`, or an interrupted run.

**In automation, don't trust the exit code of `curl … | bash`.** If curl fails, bash reads an empty script and exits 0. Download first (`curl -fsSLO …/gfv.sh && sudo bash gfv.sh …`) or run the pipe under `set -o pipefail`. A truncated download can't pass: the launcher is one `{ … }` block, so any cut after its first line is a syntax error and exits 2.

**Outputs:**
- **stdout:** one JSON result line per check.
- **Evidence directory:** `/var/lib/gpu-fleet-validation/runs/<run_id>/`, holding `SUMMARY.md`, `results.jsonl`, `junit.xml` and every raw tool output.

> **Status:** the gate has run on an 8x H100 HGX node (Ubuntu 26.04, driver 615). Its tests replay recorded output from a second 8x H100 node, and its installer is tested in Ubuntu containers. See [Status of the L2 gate](#status-of-the-l2-gate).

## Test tiers

The work is split into five test tiers. Each tier answers one question at one point in a node's life.

| Tier | Question | When | Status |
|---|---|---|---|
| L0 pod preflight | Can this training pod use its GPUs, NICs and peers? | start of every training job | planned |
| L1 health | Is any GPU, NVLink or IB link degrading? | continuously | planned |
| **L2 node acceptance** | **Is this node fit to take work?** | **node joins, repair, driver / OFED / firmware change** | **[implemented](docs/acceptance.md)** |
| L3 burn-in | Does new or RMA-returned hardware survive hours of load? | new or repaired hardware | planned (L2 has a 30-minute power soak; `--burnin-minutes` stretches it) |
| L4 cluster and fabric | Do collectives and the fabric hold up across nodes? | nightly per block, before big runs | planned |

Rules that hold across all tiers:
- **Expected values live in a node-class manifest** ([`manifests/`](manifests/)), never in test code.
- **Every check writes the same JSON result line.**
- **Fail closed:** a check that cannot run counts as a failure.

## L2 node acceptance gate

L2 checks the node against its class manifest, then puts it under load:

- **Inventory and manifest diff:** GPU count and model; driver and CUDA; GPU power limit; ECC; PCIe width; NUMA layout; kernel cmdline; Secure Boot; PCIe ACS; GPUDirect module; firmware.
- **NVLink:** link count and NVSwitch fabric state; NVSwitch count; Fabric Manager status and version.
- **RDMA:** port state and rate; the GPU-to-NIC rail map; `ib_write_bw` loopback through the switch on every rail, pinned to each NIC's NUMA node; GPUDirect RDMA over DMA-BUF or nvidia-peermem.
- **DCGM:** `dcgmi diag -r 3`, plus per-GPU PCIe bandwidth floors (one-way gates the link, bidirectional warns).
- **Burn-in:** every GPU at its power limit for 30 minutes. Checks peak GPU and HBM temperature against the manifest's thresholds, the GPU's own temperature limits, the spread between GPUs, any thermal or power-brake throttling, sustained power, and whether temperatures levelled off. See [Burn-in and temperatures](docs/acceptance.md#burn-in-and-temperatures).
- **NCCL:** single-node `all_reduce_perf`, checking bus bandwidth, correctness and NCCL version.
- **fio:** optional local-storage floors.
- **Health deltas:** XIDs, SXids, ECC, remapped rows, and IB and NVLink error counters, compared before and after the load.

**Tools it installs:** DCGM, perftest, fio and a pinned nccl-tests build. Reruns reuse them.

**What it never touches:** the GPU driver, Fabric Manager, OFED/DOCA, the kernel or the system NCCL.
- Every apt change is simulated first. It is refused if it would remove anything, pull in a driver or kernel package, or upgrade the stack under test.
- Other dependency upgrades are allowed and listed in the results.
- A temporary NVIDIA CUDA apt repo is removed again afterwards.

**Pre-staging:** to install the tools without running the checks, e.g. in an image bake, run with `--only install --allow-partial`.

**Timing:** about an hour: DCGM level 3 (about 25 minutes) and the burn-in (30 minutes). The first run adds about 5–10 minutes for installs. For a quick check, `--skip burnin --allow-partial`.

## Add a manifest for your node class

A manifest says what a healthy node of one class looks like. The runner compares the node with it, and nothing in the code is specific to one class. Start from [`manifests/hgx-h100-8x.toml`](manifests/hgx-h100-8x.toml), and read each value off a node you trust.

1. **Copy the shipped manifest** to `my-class.toml` and rename the table, e.g. `[class."hgx-h200-8x"]` and its sub-tables.
2. **Fill in each key on a known-good node:**

   | Key | Read it with |
   |---|---|
   | `gpus`, `gpu_name`, `driver`, `pcie_width`, `gpu_power_limit_w` | `nvidia-smi --query-gpu=name,driver_version,pcie.link.width.max,enforced.power.limit --format=csv` |
   | `cuda` | `nvidia-smi \| head -4`: `CUDA Version`, or `CUDA UMD Version` on driver 615+ |
   | `nccl` | `apt-cache madison libnccl2`: a version built for that CUDA major, e.g. `2.32.3` from `2.32.3-1+cuda13.4` |
   | `nvlinks_total` | `nvidia-smi nvlink -s \| grep -c GB/s` |
   | `nvswitches`, `nvswitch_pci_device` | `lspci -nn -d 10de: \| grep -i bridge`: count, and the device ID after `10de:` |
   | `rdma_nics`, `rdma_rate_gbps`, `rdma_link_layer` | `grep . /sys/class/infiniband/*/ports/1/{rate,link_layer}` |
   | `numa_nodes` | `lscpu \| grep "NUMA node(s)"` |
   | `[thermal]` | `dcgmi dmon -e 152,158,159,151 -c 1`: max operating, slowdown, shutdown and HBM max operating temperature in °C |
   | `[host]` | `cat /proc/cmdline` for `kernel_cmdline_contains`; `mokutil --sb-state` for `secure_boot`. Set `gpudirect = true` if GPUDirect RDMA must work; its test needs a CUDA-enabled perftest (DOCA or MLNX_OFED, not Ubuntu's) |

3. **Leave the firmware allowlists empty, and the `[perf]` floors out, for the first runs.** Their checks then record the measured values as `info`. Afterwards, set each floor to the lowest healthy node's value minus about 5 %. Fill the allowlists once you have decided which firmware versions you accept.
4. **Set the burn-in thresholds below the `[thermal]` limits.** The shipped class fails 5 °C under the max operating temperature and warns 10 °C under. Leave out `[burnin]` to skip the burn-in.
5. **Check the file without root or GPUs:** `bash gfv.sh accept --manifest my-class.toml --dry-run`. It validates every key; a typo is an error, not a silently skipped check.
6. **Run it:** `sudo bash gfv.sh accept --manifest my-class.toml`. A JSON manifest works on any Python; TOML needs Python 3.11+ on the node.
7. **To ship the class,** add it as `manifests/<class>.toml` and cut a release. `bash gfv.sh classes` lists it, and `--class <class>` uses it.

Every key is listed in [docs/acceptance.md](docs/acceptance.md#manifest).

## Status of the L2 gate

v0.1.0 ran on an 8x H100 HGX node with NDR InfiniBand (Ubuntu 26.04, driver 615, DCGM 4.7.0). No hardware fault was found. The script and manifest problems that run exposed are fixed; [docs/acceptance.md](docs/acceptance.md#first-hardware-run) lists them and what is still open.

The parsers and the whole runner are also tested against recorded output from a second 8x H100 HGX node, with fake tools and a fake `/sys` tree; see [tests/fixtures/README.md](tests/fixtures/README.md). The installer is tested end to end in Ubuntu containers.

**Known finding:** on both nodes, GPU 0 and GPU 1 show the lowest bidirectional PCIe bandwidth: 76 GB/s on the recorded node and 67 GB/s on the new one, against 84–101 GB/s on the other GPUs. One-way bandwidth is 53–55 GB/s on every GPU, so the links are healthy and the limit is on the host side. The shipped class therefore gates on one-way bandwidth and only warns on bidirectional. The replay of the recorded node keeps the old bidirectional-only floor and fails exactly those two checks; the tests assert that.

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
