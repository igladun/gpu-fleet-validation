# L2 node acceptance gate

L2 decides whether one node is fit to take work. Run it when:
- a node joins the cluster;
- a node comes back from repair;
- the driver, OFED/DOCA or firmware changes.

It runs on the node as root and installs the test tools it needs. The node passes only if every check passes. The gate fails closed: a check that cannot run is an `error`, and an error fails the node.

## Running it

```sh
sudo bash gfv-accept.sh accept --class hgx-h100-8x                 # bundled class manifest
sudo bash gfv-accept.sh accept --manifest my-class.json             # your own manifest
sudo bash gfv-accept.sh accept --class hgx-h100-8x --dry-run        # print the plan only
curl -fsSL https://<host>/gfv-accept.sh | sudo bash -s -- accept --class hgx-h100-8x
```

From a repo checkout, `sudo python3 -m gfv accept …` is the same thing. A checkout run needs Python 3.11+ to read the TOML manifests; the single file needs only 3.10.

| Option | Meaning |
|---|---|
| `--class NAME` | Node class: a bundled manifest, or one class inside `--manifest` |
| `--manifest FILE` | Manifest file. `.json` works everywhere; `.toml` needs Python 3.11+ |
| `--no-install` | Install nothing. A missing tool makes its checks `error` |
| `--tools-dir DIR` | Where NCCL and nccl-tests are unpacked and built. Default `/opt/gpu-fleet-validation/tools` |
| `--out DIR` | Evidence directory. Default `/var/lib/gpu-fleet-validation/runs/<run_id>` |
| `--only` / `--skip LIST` | Stages: `install,inventory,nvlink,rdma,health,dcgm,nccl,fio` |
| `--dcgm-level N` | `dcgmi diag` level: 1 quick, 2 medium, 3 full (default, about 25 min), 4 extended (hours) |
| `--force` | Run although GPU processes are running. DCGM and NCCL results are then unreliable |
| `--fail-fast` | Skip the heavy stages (RDMA bandwidth, DCGM, NCCL, fio) once a cheap check has failed |
| `--node`, `--cluster`, `--rack`, `--leaf` | Labels copied into every result line |

**Exit codes:**
- `0`: the node passed.
- `1`: at least one `fail` or `error`.
- `2`: the gate could not run. Causes: not root, a bad manifest, `nvidia-smi` not working, or GPU processes running without `--force`.

## Stages and checks

The stages run in this order. `prereq` always runs.

| Stage | Check | Pass when | Reads manifest key |
|---|---|---|---|
| prereq | `prereq.nvidia_smi` | the driver answers | – |
| prereq | `prereq.gpus_idle` | no compute processes (exit 2 otherwise) | – |
| install | `install.*` | tools present or installed (see below) | `nccl` |
| health (before) | `health.ecc_uncorrected` | volatile uncorrectable ECC = 0 (aggregate > 0 is `warn`) | – |
| health (before) | `health.remapped_rows` | no row remap pending or failed | – |
| health (before) | `health.xid` | no hardware XID since boot (48, 63, 64, 74, 79, 94, 95, 119, 120); other XIDs are `warn` | – |
| inventory | `inventory.gpu_count`, `.gpu_name` | count and model match | `gpus`, `gpu_name` |
| inventory | `inventory.driver`, `.cuda` | driver version and the CUDA version the driver reports match | `driver`, `cuda` |
| inventory | `inventory.mig` | MIG off. Persistence mode off is `warn` | – |
| inventory | `inventory.pcie_width` | every GPU and RDMA NIC runs at its max width and at the manifest width | `pcie_width` |
| inventory | `inventory.numa_nodes` | NUMA node count matches | `numa_nodes` |
| inventory | `inventory.kernel_cmdline` | each listed string appears in `/proc/cmdline` | `host.kernel_cmdline_contains` |
| inventory | `inventory.secure_boot` | `mokutil --sb-state` matches | `host.secure_boot` |
| inventory | `inventory.pcie_acs` | ACS P2P redirect (`ReqRedir`/`CmpltRedir`) on the bridges above each GPU matches | `host.pcie_acs` |
| inventory | `inventory.nvidia_peermem` | the GPUDirect RDMA module is loaded | `host.nvidia_peermem` |
| inventory | `inventory.firmware.{bios,gpu_vbios,nic}` | the value is in the allowlist; an empty list only records it (`info`) | `firmware.*` |
| inventory | `inventory.power_profile`, `.firmware.bmc` | `skip`: needs BMC/Redfish (later tier) | – |
| nvlink | `nvlink.links_total` | active links = expected; per-GPU inactive links fail | `nvlinks_total` |
| nvlink | `nvlink.fabric_state` | every GPU reports Fabric `Completed/Success` | `nvswitches` > 0 |
| nvlink | `nvlink.fabricmanager_active`, `_version` | service active, package version = driver version | `nvswitches` > 0 |
| nvlink | `nvlink.nvswitch_count` | NVSwitches on the PCI bus = expected | `nvswitches`, `nvswitch_pci_device` |
| nvlink | `nvlink.dcgm_nvswitch_visible` | `warn` when DCGM sees 0 NVSwitches. libnvidia-nscq is missing, so DCGM's NVSwitch health checks are blind | – |
| rdma | `rdma.nic_count`, `rdma.port` | RDMA devices with the expected link layer are ACTIVE / LinkUp at the expected rate | `rdma_nics`, `rdma_link_layer`, `rdma_rate_gbps` |
| rdma | `rdma.rail_map` | every GPU has its own NIC at PIX/PXB distance (`nvidia-smi topo -m`) | – |
| rdma | `rdma.write_bw` | `ib_write_bw` loopback through the switch in a ring (rail k → k+1, 5 s, 64 KiB, single QP) ≥ floor | `perf.rdma_write_bw_gbps` |
| rdma | `rdma.gdr_write_bw` | the same with `--use_cuda` on the rail-local GPUs. Advisory in v1: below the floor is `warn`; `skip` if perftest has no CUDA support | `perf.rdma_write_bw_gbps` |
| dcgm | `dcgm.diag` | every DCGM test passes on every GPU | – |
| dcgm | `dcgm.pcie_bidir` | per-GPU bidirectional PCIe bandwidth (from DCGM's pcie plugin) ≥ floor | `perf.dcgm_pcie_bidir_gbs` |
| nccl | `nccl.correctness` | 0 wrong values, `Out of bounds values : 0 OK`, rc 0 | – |
| nccl | `nccl.allreduce_busbw` | out-of-place bus bandwidth at 8 GiB ≥ floor | `perf.nccl_allreduce_busbw_gbs_1node` |
| nccl | `nccl.version` | the NCCL library version nccl-tests ran with matches | `nccl` |
| fio | `fio.seq_write`, `fio.seq_read` | sequential bandwidth with direct I/O ≥ floors (only with `[storage]`) | `storage.*` |
| health (after) | `health.ecc_delta`, `.remapped_rows`, `.xid_new` | nothing new under load | – |
| health (after) | `health.ib_counter_delta` | `symbol_error`, `link_downed` and `link_error_recovery` did not grow; other error counters growing is `warn` | – |
| health (after) | `health.nvlink_error_delta` | CRC and recovery errors did not grow; replay errors growing is `warn` | – |

When a perf floor is missing from the manifest, its check records the measured value as `info`. Use this on a new class's first healthy nodes, then set the floors from what they measured.

DCGM level 3 already runs DCGM's `nvbandwidth` plugin, so L2 does not run a separate nvbandwidth. If no DCGM host engine is running, the runner starts `nv-hostengine` for the diagnostic and stops it afterwards. It never stops an engine it did not start.

## Tool install

The install stage is on by default. It is idempotent and installs test tooling only. It never touches the driver, Fabric Manager, OFED/DOCA, the kernel or package holds.

| Need | Done only if missing |
|---|---|
| CA bundle | `apt-get install ca-certificates` (minimal images) |
| NVIDIA CUDA apt repo | `cuda-keyring` for the Ubuntu release, from developer.download.nvidia.com |
| fio, lspci, mokutil, git, make, g++, ib_write_bw | `apt-get install fio pciutils mokutil git build-essential perftest`. With a DOCA repo configured, apt picks DOCA's perftest |
| dcgmi | `apt-get install --install-recommends datacenter-gpu-manager-4-cuda<driver CUDA major>=<pinned>`. Falls back to unpinned if the pin is gone |
| NCCL | `apt-get download libnccl2 libnccl-dev`, then `dpkg-deb -x` into `<tools-dir>/nccl-<version>`. Uses the manifest's `nccl`, built for the driver's CUDA major and the highest CUDA minor not above the driver's. The system NCCL is not changed |
| nvcc | `/usr/local/cuda-<M.m>` or `/usr/local/cuda` if present, else `apt-get install cuda-nvcc-<M-m> cuda-cudart-dev-<M-m>` |
| nccl-tests | `git fetch` at the commit in [`gfv/tools.lock.json`](../gfv/tools.lock.json), then `make` against that NCCL, compiled only for the GPUs' compute capability. Reused on later runs |

`installed.json` in the evidence directory lists what the run installed. `raw/commands.log` and `raw/install-*.out` hold every install command and its output.

The automatic install supports Ubuntu (22.04, 24.04) on x86_64 and arm64 (sbsa). On anything else, install the tools yourself and pass `--no-install`.

## Results

Every check produces one JSON line. The lines go to stdout at the end of the run and to `results.jsonl`:

```json
{"ts": "2026-11-10T08:27:38Z", "run_id": "l2-20261110T082737Z-e4af", "cluster": null, "node": "gpu-node-01",
 "class": "hgx-h100-8x", "rack": null, "leaf": null, "pod": null, "tier": "L2",
 "check": "dcgm.pcie_bidir", "target": "GPU0", "value": 76.08, "expected": 95, "verdict": "fail",
 "detail": "GB/s", "versions": {"driver": "580.178.04", "cuda": "13.0", "nccl": "2.28.9", "dcgm": "4.6.1", "...": "..."},
 "evidence": "raw/dcgm-diag-r3.out"}
```

`verdict` is one of `pass`, `fail`, `warn`, `info`, `skip` or `error`. `evidence` is a path inside the evidence directory.

The evidence directory holds:

| File | What it is |
|---|---|
| `SUMMARY.md` | the verdict, versions, notes and a table with failures first |
| `results.jsonl` | every check, with the final versions on each line |
| `results.partial.jsonl` | the same lines written as they happen; survives a crash |
| `junit.xml` | for CI dashboards |
| `versions.json` | driver, CUDA, NCCL, DCGM, OFED, Fabric Manager, kernel, OS, BIOS, gfv |
| `manifest.json` | the class spec the run used |
| `installed.json` | what the install stage did |
| `raw/` | every command's stdout/stderr, plus `commands.log` with return codes and durations |
| `crash-<stage>.txt` | a traceback if a stage hit a bug. The stage is recorded as `error` and the run continues |

## Manifest

A manifest describes one node class. See [`manifests/hgx-h100-8x.toml`](../manifests/hgx-h100-8x.toml).

| Key | Type | Meaning |
|---|---|---|
| `gpus`, `gpu_name` | int, str | GPU count and `nvidia-smi` product name |
| `driver`, `cuda` | str | driver version, and the CUDA version the driver reports |
| `nccl` | str | NCCL version used for the nccl-tests build; must exist for the driver's CUDA |
| `pcie_width` | int | expected link width of every GPU and RDMA NIC |
| `nvlinks_total`, `nvswitches`, `nvswitch_pci_device` | int, int, str | NVLink count, NVSwitch count, NVSwitch PCI device ID |
| `rdma_nics`, `rdma_rate_gbps`, `rdma_link_layer` | int, int, str | rails, port rate, `InfiniBand` or `Ethernet` |
| `numa_nodes` | int | NUMA node count |
| `[host]` | table | `kernel_cmdline_contains` (list), `pcie_acs` (`disabled`/`enabled`), `secure_boot` (bool), `nvidia_peermem` (bool), `power_profile` (reported as skip) |
| `[firmware]` | table of lists | `bios`, `gpu_vbios`, `nic` allowlists; `bmc` is reported as skip |
| `[perf]` | table of numbers | `nccl_allreduce_busbw_gbs_1node`, `rdma_write_bw_gbps`, `dcgm_pcie_bidir_gbs` |
| `[storage]` | table | `path` (required), `size_gib`, `seq_write_gbs`, `seq_read_gbs` |
| `[k8s]` | table | for the in-cluster tiers; L2 ignores it |

Re-measure the floors on the first healthy nodes of each class. Never copy floors between classes.

## Runbook

1. Drain the node from the scheduler. Stop anything using the GPUs: training, inference servers, notebooks.
2. Copy the single file over, or curl it, and run it as root. Use `tmux` or `nohup` so the run survives a dropped ssh session:
   ```sh
   sudo nohup bash gfv-accept.sh accept --class hgx-h100-8x --node "$(hostname)" > accept.jsonl 2> accept.log &
   tail -f accept.log
   ```
3. Read `SUMMARY.md` in the evidence directory and archive the whole directory.
4. On exit 0, return the node to service. On exit 1, keep it out, fix what failed, and run again. Exit 2 means the gate never ran.

## First hardware run

The runner has only run against recorded output so far. The first run on a real node has to confirm the following:

- **Synthetic parsers.** `lspci -D -vvv` ACS parsing and the `--use_cuda` GPUDirect path were checked only against synthetic samples. Confirm both, and check the `mokutil` and `ofed_info` output formats.
- **Loopback ring.** `ib_write_bw` loopback uses `127.0.0.1` for the connection exchange and the IB fabric for the data. Confirm all 8 rails report bandwidth.
- **Host engine.** With the node's own DCGM service running, confirm the runner reuses it and does not start a second one.
- **GPU0/1 PCIe finding.** Settle the 76 GB/s bidirectional PCIe on GPU 0 and GPU 1. A DCGM host-memory placement artifact means adjusting the check; otherwise it is a real fault.
- **Manifest values.** Re-measure the floors. Set `secure_boot`, `pcie_acs` and `kernel_cmdline_contains` to what the class really runs, then fill the firmware allowlists from the recorded values.
