# L2 node acceptance gate

L2 decides whether one node is fit to take work. Run it when:
- a node joins the cluster;
- a node comes back from repair;
- the driver, OFED/DOCA or firmware changes.

It runs on the node as root and installs the test tools it needs. The node passes only if every check passes. The gate fails closed: a check that cannot run is an `error`, and an error fails the node.

L2 checks the node against its class [manifest](manifest.md), then puts it under load:

- **Inventory and manifest diff:** GPU count and model; driver and CUDA; GPU power limit; ECC; PCIe width; NUMA layout; kernel cmdline; Secure Boot; PCIe ACS; GPUDirect module; firmware.
- **NVLink:** link count and NVSwitch fabric state; NVSwitch count; Fabric Manager status and version.
- **RDMA:** port state and rate; the GPU-to-NIC rail map; `ib_write_bw` loopback through the switch on every rail, pinned to each NIC's NUMA node; GPUDirect RDMA over DMA-BUF or nvidia-peermem.
- **DCGM:** `dcgmi diag -r 3`, plus per-GPU PCIe bandwidth floors (one-way gates the link, bidirectional warns).
- **Burn-in:** every GPU at its power limit for 30 minutes. Checks peak GPU and HBM temperature against the manifest's thresholds, the GPU's own temperature limits, the spread between GPUs, any thermal or power-brake throttling, sustained power, and whether temperatures levelled off. See [Burn-in and temperatures](#burn-in-and-temperatures).
- **NCCL:** single-node `all_reduce_perf`, checking bus bandwidth, correctness and NCCL version.
- **fio:** optional local-storage floors.
- **Health deltas:** XIDs, SXids, ECC, remapped rows, and IB and NVLink error counters, compared before and after the load.

[Stages and checks](#stages-and-checks) lists every check.

## Running it

Run it on the GPU node itself. It needs Ubuntu 22.04, 24.04 or 26.04 and a working NVIDIA driver; it installs everything else. Stop GPU workloads first; the gate refuses to run while GPU processes exist.

Each [release](https://github.com/igladun/gpu-fleet-validation/releases) has three files:
- `gfv.sh`: a short launcher you can read in a minute. It downloads `gfv.pyz` from the same release, checks its sha256 and runs it with `python3`. It installs `python3` first if the node lacks it.
- `gfv.pyz`: the code, a Python zipapp built from [`gfv/`](../gfv/) and [`manifests/`](../manifests/). List it with `python3 -m zipfile -l gfv.pyz`.
- `SHA256SUMS`.

```sh
# latest release
curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | sudo bash -s -- accept --class hgx-h100-8x

# or download the launcher once, then run
curl -fsSLO https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh
sudo bash gfv.sh accept --class hgx-h100-8x                 # bundled class manifest
sudo bash gfv.sh accept --manifest my-class.toml             # your own manifest, see manifest.md
bash gfv.sh accept --class hgx-h100-8x --dry-run             # print the plan only: no root, checks and changes nothing
bash gfv.sh classes                                          # list the bundled classes
bash gfv.sh accept --help                                    # every option
```

To keep a run alive over ssh, start it under `nohup` or `tmux`; see the [Runbook](#runbook).

**Timing:** about an hour: DCGM level 3 (about 25 minutes) and the burn-in (30 minutes). The first run adds about 5–10 minutes for installs. For a quick check, `--skip burnin --allow-partial`.

**Pin a version:** replace `latest/download` with `download/<tag>`, e.g. `download/v0.1.0`. Each `gfv.sh` carries the version and sha256 of its own `gfv.pyz`.

**Verify a download:** `sha256sum -c --ignore-missing SHA256SUMS`, or `gh attestation verify gfv.pyz -R igladun/gpu-fleet-validation` to check that the release workflow built it from this repo.

**When the node can't reach GitHub**, copy `gfv.sh` and `gfv.pyz` from the same release to the node. `GFV_PYZ` makes the launcher use the local copy, and it still checks the sha256 it carries:

```sh
sudo GFV_PYZ="$PWD/gfv.pyz" bash gfv.sh accept --class hgx-h100-8x
```

The install stage still needs apt, NVIDIA's CUDA repo and github.com for nccl-tests; see [Tool install](#tool-install). Where those are blocked, install the tools yourself and add `--no-install`.

From a repo checkout, `sudo python3 -m gfv accept …` is the same thing. A checkout run needs Python 3.11+ to read the TOML manifests; `gfv.pyz` bundles them as JSON and needs only 3.10.

| Option | Meaning |
|---|---|
| `--class NAME` | Node class: a bundled manifest, or one class inside `--manifest` |
| `--manifest FILE` | Manifest file. `.json` works everywhere; `.toml` needs Python 3.11+ |
| `--no-install` | Install nothing. A missing tool makes its checks `error` |
| `--tools-dir DIR` | Where NCCL and nccl-tests are unpacked and built. Default `/opt/gpu-fleet-validation/tools` |
| `--out DIR` | Evidence directory. Default `/var/lib/gpu-fleet-validation/runs/<run_id>` |
| `--only` / `--skip LIST` | Stages: `install,inventory,nvlink,rdma,health,dcgm,burnin,nccl,fio` |
| `--dcgm-level N` | `dcgmi diag` level: 1 quick, 2 medium, 3 full (default, about 25 min), 4 extended (hours) |
| `--burnin-minutes N` | Burn-in length at the power limit. Default: the manifest's `burnin.minutes` (30 for `hgx-h100-8x`). Use 240 or more for a first intake soak |
| `--force` | Run although GPU processes are running. DCGM and NCCL results are then unreliable |
| `--fail-fast` | Skip the heavy stages (RDMA bandwidth, DCGM, burn-in, NCCL, fio) once a cheap check has failed |
| `--allow-partial` | Let a run with `--only` or `--skip` pass. Without it, a partial run fails (`run.partial`) so it can never certify a node. To pre-stage tools, e.g. in an image bake: `--only install --allow-partial` |
| `--node`, `--cluster`, `--rack`, `--leaf` | Labels copied into every result line |

**Exit codes:**
- `0`: the node passed.
- `1`: at least one `fail` or `error`.
- `2`: the gate could not run. Causes:
  - the launcher could not download `gfv.pyz`, or its sha256 did not match
  - not root
  - a bad manifest, or no stages selected
  - a non-empty `--out`
  - `nvidia-smi` not working
  - the GPU process list cannot be read
  - GPU processes running without `--force`
  - the run was interrupted (SIGTERM, SIGHUP, Ctrl-C) before it finished

**In automation, don't trust the exit code of `curl … | bash`.** If curl fails, bash reads an empty script and exits 0. Download first (`curl -fsSLO …/gfv.sh && sudo bash gfv.sh …`) or run the pipe under `set -o pipefail`, and read `results.jsonl`, not only the exit code. A truncated download can't pass: the launcher is one `{ … }` block, so any cut after its first line is a syntax error and exits 2.

A partial run (`--only`/`--skip`) exits `1` unless `--allow-partial` is given. Every stage it left out is recorded as `<stage>.not_selected`.

## Stages and checks

The stages run in this order. `prereq` always runs.

| Stage | Check | Pass when | Reads manifest key |
|---|---|---|---|
| prereq | `prereq.nvidia_smi` | the driver answers | – |
| prereq | `prereq.gpus_idle` | the process list can be read and is empty (exit 2 otherwise) | – |
| install | `install.*` | tools present or installed (see below). `install.cuda_version` is `error` when the driver's CUDA version cannot be read; DCGM and NCCL installs depend on it | `nccl` |
| health (before) | `health.ecc_uncorrected` | volatile uncorrectable ECC = 0 on all GPUs (aggregate > 0 is `warn`); unreadable counters or a short GPU list are `error` | `gpus` |
| health (before) | `health.remapped_rows` | no row remap pending or failed | – |
| health (before) | `health.xid` | no XID since boot. Only application XIDs 13, 31, 43, 45 are `warn`; every other XID fails | – |
| health (before) | `health.sxid` | NVSwitch SXids since boot are `warn` | `nvswitches` > 0 |
| health (before) | `health.ib_counter` | a saturated error counter (e.g. `symbol_error` 65535) fails, because new errors can no longer be seen; other non-zero counters are `warn` | – |
| health (before) | `health.ib_counters_readable`, `health.nvlink_counters_readable` | every rail exposes its error counters, and `nvidia-smi nvlink -e` reports every link | `nvlinks_total` |
| inventory | `inventory.gpu_count`, `.gpu_name` | count and model match | `gpus`, `gpu_name` |
| inventory | `inventory.driver`, `.cuda` | driver version and the CUDA version the driver reports match. Both banner layouts are read: `Driver Version` / `CUDA Version` (up to R580) and `KMD Version` / `CUDA UMD Version` (R615). The driver falls back to `--query-gpu=driver_version`. A value that cannot be read is `error` | `driver`, `cuda` |
| inventory | `inventory.power_limit` | every GPU's enforced power limit equals the class value. A lower value means someone capped the GPU | `gpu_power_limit_w` |
| inventory | `inventory.mig` | MIG off. Persistence mode off is `warn` | – |
| inventory | `inventory.ecc_mode` | ECC enabled on every GPU | – |
| inventory | `inventory.pcie_width` | every GPU and RDMA NIC runs at its max width and at the manifest width | `pcie_width` |
| inventory | `inventory.numa_nodes` | NUMA node count matches | `numa_nodes` |
| inventory | `inventory.kernel_cmdline` | each listed string appears in `/proc/cmdline` | `host.kernel_cmdline_contains` |
| inventory | `inventory.secure_boot` | `mokutil --sb-state` matches | `host.secure_boot` |
| inventory | `inventory.pcie_acs` | ACS P2P redirect (`ReqRedir`/`CmpltRedir`) on the PCIe switch ports above each GPU and each RDMA NIC matches. Root ports are not checked. No checkable port at all is `error` | `host.pcie_acs` |
| inventory | `inventory.nvidia_peermem` | with `host.nvidia_peermem` set: the module state matches. Without it the state is recorded as `info`; set `host.gpudirect = true` to require a working GPUDirect test instead | `host.nvidia_peermem` |
| inventory | `inventory.firmware.{bios,gpu_vbios,nic}` | the value is in the allowlist; an empty list only records it (`info`) | `firmware.*` |
| inventory | `inventory.power_profile`, `.firmware.bmc` | `skip`: needs BMC/Redfish (later tier) | – |
| nvlink | `nvlink.links_total` | active links = expected; per-GPU inactive links fail | `nvlinks_total` |
| nvlink | `nvlink.fabric_state` | every GPU reports Fabric `Completed/Success` | `nvswitches` > 0 |
| nvlink | `nvlink.fabricmanager_active`, `_version` | service active, package version = driver version. An installed package on hold (`hi`) counts | `nvswitches` > 0 |
| nvlink | `nvlink.nvswitch_count` | NVSwitches on the PCI bus = expected | `nvswitches`, `nvswitch_pci_device` |
| nvlink | `nvlink.dcgm_nvswitch_visible` | `warn` when DCGM sees 0 NVSwitches. libnvidia-nscq is missing, so DCGM's NVSwitch health checks are blind | – |
| rdma | `rdma.nic_count`, `rdma.port` | RDMA devices with the expected link layer are ACTIVE / LinkUp at the expected rate | `rdma_nics`, `rdma_link_layer`, `rdma_rate_gbps` |
| rdma | `rdma.rail_map` | every GPU has its own NIC at PIX/PXB distance (`nvidia-smi topo -m`); a matrix missing GPUs is `error` | `gpus` |
| rdma | `rdma.write_bw` | `ib_write_bw` loopback through the switch in a ring (rail k → k+1, 5 s, 64 KiB, single QP) ≥ floor. Server and client run under `taskset` on each NIC's local CPUs (`/sys/class/infiniband/<dev>/device/local_cpulist`), so their buffers sit on the NIC's NUMA node; `detail` says whether the pair was pinned | `perf.rdma_write_bw_gbps` |
| rdma | `rdma.gdr_write_bw` | the same with `--use_cuda` on the rail-local GPUs. DMA-BUF (`--use_cuda_dmabuf`, open kernel module) is tried first, then the nvidia-peermem path; the first path that works is used for the remaining pairs and named in `detail`. GPU memory that no path can register is `error`. Below the floor is `warn` until a floor is measured. Without CUDA support in perftest (Ubuntu's own perftest has none; DOCA/MLNX_OFED perftest has) the check is `skip`, or `error` when `host.gpudirect = true` | `perf.rdma_write_bw_gbps`, `host.gpudirect` |
| dcgm | `dcgm.diag` | every test passes on every GPU. Each test's summary must not be Fail. Each test must cover all GPUs. At level 3+ the tests software, memory, diagnostic, nvbandwidth, pcie, targeted_stress and targeted_power must all run (a missing one, or Skip, is `error`). A non-zero `dcgmi` exit with nothing failing in the JSON is `error`. DCGM 4.x is required | `gpus` |
| dcgm | `dcgm.pcie_oneway` | per-GPU PCIe bandwidth in each direction (GPU to host, host to GPU) ≥ floor. This is the link-health gate: a healthy Gen5 x16 link sits near 55 GB/s each way | `perf.dcgm_pcie_oneway_gbs` |
| dcgm | `dcgm.pcie_bidir` | per-GPU bidirectional PCIe bandwidth ≥ floor. Bidirectional also depends on the host (IOMMU mode, NUMA layout, root complex), so below the floor it is `warn` when that GPU's one-way check passes, and `fail` otherwise (or when no one-way floor is set) | `perf.dcgm_pcie_bidir_gbs` |
| burnin | `burnin.limits` | every GPU reports its temperature limits (DCGM fields 152, 158, 159, 151) and they equal the manifest's; a mismatch means another SKU or VBIOS. A limit the manifest sets but a GPU does not report is `error` | `thermal.*` |
| burnin | `burnin.dcgm` | `dcgmi diag -r targeted_power` held for `burnin.minutes` passes on every GPU | `burnin.minutes` |
| burnin | `burnin.telemetry` | per GPU, the samples under load (power ≥ 80 % of `gpu_power_limit_w`) cover ≥ 90 % of the requested minutes. Less means the load or the monitor stopped early: `error` | `burnin.minutes` |
| burnin | `burnin.gpu_temp` | per GPU, the hottest reading ≤ `gpu_temp_max_c` (`fail` above); above `gpu_temp_warn_c` is `warn`. `detail` gives the margin to the max operating temperature | `burnin.gpu_temp_max_c`, `.gpu_temp_warn_c` |
| burnin | `burnin.mem_temp` | per GPU, the hottest HBM reading ≤ `mem_temp_max_c` | `burnin.mem_temp_max_c` |
| burnin | `burnin.power` | per GPU, the median power under load after ramp-up ≥ `power_min_w` | `burnin.power_min_w` |
| burnin | `burnin.temp_spread` | hottest minus coolest GPU (median of the last 5 min) ≤ `gpu_temp_spread_c`, else `warn`: one GPU much hotter than its neighbours points at a fan or heatsink zone | `burnin.gpu_temp_spread_c` |
| burnin | `burnin.steady_state` | no GPU still warming faster than 0.5 °C/min over the last 5 min under load, else `warn` (extend the burn-in). The cool-down after the load is left out. Runs shorter than 10 min record `info` | – |
| burnin | `burnin.throttle` | the thermal-slowdown and power-brake counters (`nvidia-smi … clocks_event_reasons_counters.*`) did not grow during the burn-in. Counters that cannot be read or are `N/A` are `error` | – |
| nccl | `nccl.correctness` | 0 wrong values, `Out of bounds values : 0 OK`, rc 0 | – |
| nccl | `nccl.allreduce_busbw` | out-of-place bus bandwidth at 8 GiB ≥ floor; a log that stops before 8 GiB is `error` | `perf.nccl_allreduce_busbw_gbs_1node` |
| nccl | `nccl.version` | the NCCL library version nccl-tests ran with matches | `nccl` |
| fio | `fio.seq_write`, `fio.seq_read` | sequential bandwidth with direct I/O ≥ floors (only with `[storage]`); a fio error or 0 bandwidth is `error` | `storage.*` |
| health (after) | `health.ecc_delta`, `.remapped_rows`, `.xid_new`, `.sxid_new` | nothing new under load (any new SXid fails) | – |
| health (after) | `health.ib_counter_delta` | `symbol_error`, `link_downed` and `link_error_recovery` did not grow; other error counters growing is `warn` | – |
| health (after) | `health.nvlink_error_delta` | CRC and recovery errors did not grow; replay errors growing is `warn` | – |

When a perf floor is missing from the manifest, its check records the measured value as `info`. Use this on a new class's first healthy nodes, then set the floors from what they measured.

DCGM level 3 already runs DCGM's `nvbandwidth` plugin, so L2 does not run a separate nvbandwidth. If no DCGM host engine is running, the runner starts `nv-hostengine` for the diagnostic and stops it afterwards. It never stops an engine it did not start.

## Tool install

The install stage is on by default. It is idempotent and installs test tooling only. It never touches the driver, Fabric Manager, OFED/DOCA, the kernel or package holds:

- **Every apt transaction is simulated first** (`apt-get -s`). The install is refused, as an `error`, if apt would:
  - remove any package;
  - pull in, even as a dependency, any NVIDIA package (`nvidia-*`, `libnvidia-*`), the CUDA driver or compat packages, `dkms`, a kernel package, the system NCCL, or OFED/DOCA. A new `nvidia-utils-*`, for instance, would overwrite the `nvidia-smi` of a driver installed from NVIDIA's `.run` file;
  - upgrade any part of the stack under test: NVIDIA packages, CUDA driver/compat, OFED/DOCA/mlnx, RDMA userspace (`rdma-core`, `libibverbs`, …), the system NCCL or DCGM.

  Any other upgrade apt needs for a dependency is allowed and recorded as an `install.upgraded` warning listing every `package old -> new`. A typical case is a patch-level `perl-base` bump pulled in by `build-essential`. Package maintainer scripts can restart their own service when upgraded (an `openssh-server` upgrade restarts sshd), so review that list.

  New dependency packages are listed in `install.new_dependencies`. New RDMA userspace (`rdma-core`, `libibverbs1`, …) is allowed so perftest works on inbox-driver hosts, but flagged as `warn`.
- **Installs use `--no-remove` and `--no-install-recommends`.** `NEEDRESTART_MODE=l`, so no service is restarted.
- **The CUDA repo is temporary.** If the node had no NVIDIA CUDA apt repo, `cuda-keyring` is downloaded, checked against the sha256 in `tools.lock.json`, and purged again at the end of the stage. The keyring ships an apt pin (priority 600) that would otherwise make later upgrades prefer NVIDIA's driver packages. A repo the node already had is left alone. A `cuda-keyring` package installed without its apt source is reported as an `error` and never touched.

| Need | Done only if missing |
|---|---|
| CA bundle | `apt-get install ca-certificates` (minimal images) |
| NVIDIA CUDA apt repo | `cuda-keyring` for the Ubuntu release, from developer.download.nvidia.com |
| fio, lspci, mokutil, git, make, g++, ib_write_bw | `apt-get install fio pciutils mokutil git build-essential perftest`. With a DOCA repo configured, apt picks DOCA's perftest |
| dcgmi | `apt-get install datacenter-gpu-manager-4-cuda<M> datacenter-gpu-manager-4-proprietary-cuda<M> datacenter-gpu-manager-4-proprietary` at the pinned version (`<M>` = the driver's CUDA major), plus `libnuma1`. Falls back to unpinned if the pin is gone |
| NCCL | `apt-get download libnccl2 libnccl-dev`, then `dpkg-deb -x` into `<tools-dir>/nccl-<version>`. Uses the manifest's `nccl`, built for the driver's CUDA major and the highest CUDA minor not above the driver's. The system NCCL is not changed |
| nvcc | `/usr/local/cuda-<M.m>`, or `/usr/local/cuda` if its `version.json` has the same CUDA major, else `apt-get install cuda-nvcc-<M-m> cuda-cudart-dev-<M-m>` |
| nccl-tests | `git fetch` at the commit in [`gfv/tools.lock.json`](../gfv/tools.lock.json), then `make` against that NCCL, compiled only for the GPUs' compute capability. Reused on later runs |

`installed.json` in the evidence directory lists what the run installed. `raw/commands.log` and `raw/install-*.out` hold every install command and its output.

The automatic install supports Ubuntu (22.04, 24.04, 26.04) on x86_64 and arm64 (sbsa). On anything else, install the tools yourself and pass `--no-install`.

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

Some lines describe the run itself:
- `run.stages`: the stages that ran.
- `run.partial`: the stages left out.
- `run.test_overrides`: a `warn` when a test-only `GFV_*` variable is set.
- `run`: interrupted, e.g. by SIGTERM or SIGHUP. Cleanup still runs: the perftest server, a host engine the run started, and the fio data file are all removed.

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
| `crash-run.txt` | a traceback if the runner itself crashed outside a stage. Recorded as `run.crash` |

## Manifest

The manifest keys, and how to write a manifest for a new node class, are in [manifest.md](manifest.md).

## Burn-in and temperatures

DCGM level 3 already drives every GPU to its power limit, but only for a couple of minutes per test, so temperatures are still climbing when it stops. The `burnin` stage holds the limit long enough to reach steady state. It runs `dcgmi diag -r targeted_power` for `burnin.minutes`, samples temperatures, power and SM clock every 5 s with `dcgmi dmon`, and reads the throttle counters before and after. Power is heat, so this is the worst thermal case the GPUs see in service. The GPUs cannot be loaded harder than their power limit, only for longer.

Limits of the `hgx-h100-8x` reference node, as DCGM reports them (identical on all 8 GPUs):

| Limit | Value | What happens |
|---|---|---|
| GPU max operating temperature | 87 °C | rated ceiling for continuous work |
| GPU slowdown | 89 °C | the hardware throttles clocks |
| GPU shutdown | 95 °C | the GPU powers off |
| HBM max operating temperature | 95 °C | |

Driver 615 no longer prints absolute limits in `nvidia-smi -q`, only margins (`T.Limit`). DCGM fields 152, 158, 159 and 151 still give absolute values, so the gate reads those.

The shipped thresholds keep a margin under those limits:
- `fail` above 82 °C, which is 5 °C under max operating;
- `warn` above 77 °C, which is 10 °C under;
- HBM `fail` above 85 °C;
- `warn` when one GPU runs more than 15 °C hotter than the coolest.

GPU temperatures follow the inlet air temperature, which only the BMC reports (a later tier), so compare nodes in the same row.

If the run is interrupted, `dcgmi diag` is killed, but a host engine the node already ran may finish the targeted_power test on its own, up to `burnin.minutes`. A rerun then refuses to start while the GPUs are busy (`prereq.gpus_idle`). The monitor (`dcgmi dmon -c`) stops by itself.

The evidence directory gets `burnin-temperatures.csv` (seconds, then one column per GPU) for plotting. 30 minutes is enough to reach steady state. For new or repaired hardware, run a longer soak with `--burnin-minutes 240` or more. A multi-hour burn-in with mixed loads is tier L3.

## Runbook

1. Drain the node from the scheduler. Stop anything using the GPUs: training, inference servers, notebooks.
2. Download the launcher and run it as root. Use `tmux` or `nohup` so the run survives a dropped ssh session. Under `nohup`, SIGHUP stays ignored; any other SIGHUP or SIGTERM stops the run cleanly with exit 2:
   ```sh
   curl -fsSLO https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh
   sudo nohup bash gfv.sh accept --class hgx-h100-8x --node "$(hostname)" > accept.jsonl 2> accept.log &
   tail -f accept.log
   ```
3. Read `SUMMARY.md` in the evidence directory and archive the whole directory.
4. On exit 0, return the node to service. On exit 1, keep it out, fix what failed, and run again. Exit 2 means the gate never ran.

## Status

v0.2.0 passed a full run on an 8x H100 HGX node in 58 minutes:
- **Burn-in:** 30 minutes at 699 W per GPU, with temperatures level at 60–73 °C, 14–27 °C under the 87 °C limit, and no throttling.
- **NCCL:** all_reduce reached 479.8 GB/s.
- **RDMA:** 365–367 Gb/s on every rail.
- **GPUDirect RDMA:** only 110–205 Gb/s, flagged as a warning; most likely the node's IOMMU in translated mode.

See [First hardware run](#first-hardware-run).

The parsers and the whole runner are also tested against recorded output from a second 8x H100 HGX node, with fake tools and a fake `/sys` tree; see [tests/fixtures/README.md](../tests/fixtures/README.md). That replay keeps the old bidirectional-only PCIe floor, so it fails exactly the GPU 0 and GPU 1 bidirectional checks, and the tests assert that. The installer is tested end to end in Ubuntu containers.

## First hardware run

v0.1.0, then v0.2.0, ran on an 8x H100 SXM HGX node with 8 NDR rails, on Ubuntu 26.04, driver 615 (open kernel module) and DCGM 4.7.0. No hardware fault was found.

The full v0.2.0 run took 58 min and passed: 100 checks passed, 9 warnings, 0 fail. Its measurements:

| What | Measured | Floor or limit |
|---|---|---|
| Burn-in, 30 min at the power limit | 698.6–699.1 W median per GPU; no thermal slowdown or power brake | ≥ 665 W |
| GPU temperature under load | levels off within 1 minute at 60–73 °C (slope 0.02 °C/min) | 87 °C max operating, fail above 82 °C |
| HBM temperature under load | 55–65 °C | 95 °C max operating, fail above 85 °C |
| Spread between GPUs | 13 °C (GPU7 73 °C, GPU6 60 °C) | warn above 15 °C |
| NCCL all_reduce, 8 GiB, 8 GPUs | 479.8 GB/s, NCCL 2.32.3 | ≥ 455 |
| RDMA loopback, NUMA-pinned | 365–367 Gb/s on all 8 rails | ≥ 350 |
| PCIe one-way, per GPU | 53.9–55.3 GB/s | ≥ 50 |
| PCIe bidirectional, per GPU | 67–68 (GPU0/1), 84–88 (GPU2–7) GB/s | warn below 65 |
| GPUDirect RDMA over DMA-BUF | 110 Gb/s (both GPUs under one PCIe switch), 205 Gb/s (GPUs on different sockets) | `warn` below 350 |

GPUDirect is the one weak result. On one rail pair, writing into GPU memory reached 191 Gb/s, reading from GPU memory 139 Gb/s, and host memory on both sides 366 Gb/s. The device and pinned memory types both gave the same result. The node boots with the IOMMU in translated mode (no `iommu=pt`), which routes NIC↔GPU peer traffic through the root complex, and the pairs that share a PCIe switch are slowest. Confirming that needs a reboot with `iommu=pt`. Until then a GPUDirect floor stays unset, and the check warns.

**Confirmed on hardware:**
- **ACS parsing:** ACS controls are readable on the 24 PCIe switch ports above the 8 GPUs and 8 NICs.
- **Output formats:** `mokutil --sb-state` and `ofed_info -s` parse as expected.
- **Loopback ring:** all 8 rails report bandwidth.
- **DCGM:** the 4.7 level-3 test names are unchanged, and the runner reused the node's own host engine.
- **Burn-in tooling:** `targeted_power.test_duration` is honoured (1800 s requested, 1812 s run). `dcgmi dmon` columns follow the `-e` order. The `clocks_event_reasons_counters.*` fields read on driver 615.
- **nccl-tests 2.21.1** builds and runs against NCCL 2.32.3.
- **PCIe pattern:** GPU 0 and GPU 1 have the lowest bidirectional PCIe bandwidth on both nodes (67 GB/s here, 76 GB/s on the first reference node). One-way bandwidth is 53–55 GB/s on every GPU of both nodes, so the links are healthy and the dip is host-side. The check now gates on one-way bandwidth (`dcgm.pcie_oneway`).

**Fixed after v0.1.0:**
- **Driver 615 banner:** `nvidia-smi` prints `KMD Version` and `CUDA UMD Version`. The parser reads both layouts and falls back to the per-GPU driver version. An unreadable value is now `error`, never a `fail` with an empty value.
- **Fabric Manager on hold:** the package is installed but held (`hi`). It now counts as installed.
- **GPUDirect:** `nvidia_peermem` was not loaded, so the peermem path could not register GPU memory (`error=14`). The GPUDirect check now tries DMA-BUF first. With `host.gpudirect = true`, the test must run.
- **NCCL pin:** the pinned NCCL 2.28.9 does not exist in NVIDIA's Ubuntu 26.04 repo, which has 2.30.7–2.32.3. The manifest now pins 2.32.3, and the install error lists what the repo offers.
- **RDMA placement:** two RDMA pairs measured 327 and 335 Gb/s on an unpinned run. Pinned to the NIC's local CPUs, all 8 measure 365–367 Gb/s.
- **Manifest:** the manifest now describes this node: 4 NUMA nodes, IOMMU in translated mode, Secure Boot off, no `nvidia_peermem` requirement. It also adds the power limit, the temperature limits and the burn-in.

**Still open:**
- Signals through sudo: start with `sudo nohup bash gfv.sh …`, send SIGTERM to `sudo`, and expect exit 2.
- A GPUDirect floor, once the IOMMU question is settled.
- `libnvidia-nscq` is missing on the node, so DCGM cannot see the NVSwitches (`warn`).
