# Node-class manifest

A manifest says what a healthy node of one class looks like. The runner compares the node with it, and nothing in the code is specific to one class. Pick a bundled class with `--class <name>` (list them with `bash gfv.sh classes`), or pass your own file with `--manifest my-class.toml`.

The shipped class is [`manifests/hgx-h100-8x.toml`](../manifests/hgx-h100-8x.toml).

## Add a manifest for your node class

Start from the shipped manifest, and read each value off a node you trust.

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

Re-measure the floors on the first healthy nodes of each class. Never copy floors between classes.

## Keys

| Key | Type | Meaning |
|---|---|---|
| `gpus`, `gpu_name` | int, str | GPU count and `nvidia-smi` product name |
| `driver`, `cuda` | str | driver version, and the CUDA version the driver reports |
| `nccl` | str | NCCL version used for the nccl-tests build; must exist for the driver's CUDA |
| `pcie_width` | int | expected link width of every GPU and RDMA NIC |
| `nvlinks_total`, `nvswitches`, `nvswitch_pci_device` | int, int, str | NVLink count, NVSwitch count, NVSwitch PCI device ID |
| `rdma_nics`, `rdma_rate_gbps`, `rdma_link_layer` | int, int, str | rails, port rate, `InfiniBand` or `Ethernet` |
| `numa_nodes` | int | NUMA node count |
| `gpu_power_limit_w` | int | enforced power limit of every GPU in W (optional) |
| `[host]` | table | `kernel_cmdline_contains` (list), `pcie_acs` (`disabled`/`enabled`), `secure_boot` (bool), `nvidia_peermem` (bool, optional), `gpudirect` (bool, optional: the GPUDirect RDMA test must run and work), `power_profile` (reported as skip) |
| `[firmware]` | table of lists | `bios`, `gpu_vbios`, `nic` allowlists; `bmc` is reported as skip |
| `[thermal]` | table of numbers | the GPU's own limits in °C: `gpu_max_operating_c`, `gpu_slowdown_c`, `gpu_shutdown_c`, `mem_max_operating_c` |
| `[burnin]` | table | `minutes` (int), and thresholds `power_min_w`, `gpu_temp_warn_c`, `gpu_temp_max_c`, `mem_temp_max_c`, `gpu_temp_spread_c`. A value a threshold needs but the GPU does not report is `error`; leave the threshold out to skip that check. Without the table the burn-in stage is skipped, and `--burnin-minutes` is an `error` |
| `[perf]` | table of numbers | `nccl_allreduce_busbw_gbs_1node`, `rdma_write_bw_gbps`, `dcgm_pcie_oneway_gbs`, `dcgm_pcie_bidir_gbs` |
| `[storage]` | table | `path` (required), `size_gib`, `seq_write_gbs`, `seq_read_gbs` |
| `[k8s]` | table | for the in-cluster tiers; L2 ignores it |

Which check reads which key is listed in [acceptance.md](acceptance.md#stages-and-checks).
