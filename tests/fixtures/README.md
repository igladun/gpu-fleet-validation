# Test fixtures

`h100-8x-ib/` is one 8x H100 SXM HGX node with eight NDR InfiniBand rails (ConnectX-7), driver
580.178.04, CUDA 13.0, DCGM 4.6.1, NCCL 2.28.9. The fake tools in `../fakebin/` serve these files
in place of the real commands.

The fixtures fall into three kinds:

- **Recorded:** real tool output from a certification run. Only the parts the parsers read are kept.
- **Derived:** values from that run, reshaped into the format the runner's own commands produce.
- **Synthetic:** written by hand because the run did not record it. A synthetic file only
  proves that its parser handles the documented format. The first hardware run must confirm it.

Scrub applied to every recorded or derived file:
- The hostname is `gpu-node-01`.
- GPU UUIDs are `GPU-00000000-0000-4000-8000-00000000000N`.
- GPU serials are `16500000000NN`.
- IB GUIDs, MACs, IP addresses and site names are not included.

| File | Kind | What it is |
|---|---|---|
| `nvidia-smi.txt` | recorded | plain `nvidia-smi` banner and GPU table |
| `gpus.csv` | derived | `--query-gpu` CSV in the runner's column order, built from `nvidia-smi -q` and a query CSV |
| `gpus.fields` | derived | the exact field list the runner must request (the fake tool rejects any other) |
| `remapped.csv` | recorded | `--query-remapped-rows` |
| `compute-apps.csv` | recorded | `--query-compute-apps` with no processes (empty) |
| `topo.txt` | recorded | `nvidia-smi topo -m`, with ANSI underline codes, 8 IB rails + 11 Ethernet NICs |
| `nvlink-s.txt` | recorded | `nvidia-smi nvlink -s`, 18 links per GPU |
| `nvlink-e.txt` | recorded | `nvidia-smi nvlink -e` before the load |
| `nvlink-e-after.txt` | recorded | `nvidia-smi nvlink -e` after the load |
| `smi-q.txt` | derived | `nvidia-smi -q` cut down to the GPU headers and Fabric sections |
| `lspci-nvidia.txt` | recorded | `lspci -nn -d 10de:`, 8 GPUs + 4 NVSwitches (`22a3`) |
| `dcgmi-discovery.txt` | recorded | `dcgmi discovery -l`, which reports 0 NvSwitches (libnvidia-nscq was missing) |
| `dcgmi-version.txt` | recorded | `dcgmi --version` |
| `dcgm-diag-r3.json` | recorded | `dcgmi diag -r 3 -j`; GPU0/1 show 76 GB/s bidirectional PCIe, GPU2-7 show 101 |
| `all_reduce_perf.log` | recorded | nccl-tests `all_reduce_perf -b 8 -e 8G -f 2 -g 8`, 478.6 GB/s busbw at 8 GiB |
| `rdma-bw.json` | derived | per-client `ib_write_bw` loopback averages (Gb/s) from the recorded run |
| `dmesg-before.txt` | recorded | NVIDIA kernel messages before the load (no XID) |
| `dmesg-after.txt` | recorded | NVIDIA kernel messages after the load (no XID) |
| `fabricmanager-dpkg.txt` | recorded | the nvidia-fabricmanager package version |
| `systemctl-fabricmanager.txt` | derived | `systemctl is-active` from the recorded service status |
| `sysfs.json` | derived | fake `/sys`, `/proc` and `/etc` tree. IB port state, rate, link layer, firmware and counters are recorded; PCI paths, BIOS version, kernel cmdline and module list are synthetic |
| `fio-write.json` | recorded | fio 3.36 sequential write, generated in a container (not on the GPU node) |
| `fio-read.json` | recorded | fio 3.36 sequential read, generated in a container (not on the GPU node) |
| `lspci-vvv.txt` | synthetic | `lspci -D -vvv` ACS controls on the bridges above each GPU; **not verified on hardware** |
| `mokutil.txt` | synthetic | `mokutil --sb-state` |
| `ofed-info.txt` | synthetic | `ofed_info -s` |
| `ib-write-bw-help.txt` | synthetic | perftest help text without `--use_cuda` (no GPUDirect build) |

Also in this directory:
- `manifest-hgx-h100-8x.json` is a JSON copy of `manifests/hgx-h100-8x.toml`. The Python 3.10 tests use it because 3.10 has no TOML parser. A test checks that the copy matches the TOML.
- `install/apt-madison.txt` holds the `libnccl2` versions published in the NVIDIA CUDA apt repo for Ubuntu 24.04, in `apt-cache madison` format.
