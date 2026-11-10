# gpu-fleet-validation

Scripts that validate multi-node GPU training clusters: 8-GPU HGX-class nodes with one RDMA NIC per GPU, on an InfiniBand or RoCE fabric.

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

L2 runs on the node itself, as root. It checks the node against its class manifest, then puts it under load:

- **Inventory and manifest diff:** GPU count and model; driver and CUDA; PCIe width; NUMA layout; kernel cmdline; Secure Boot; PCIe ACS; GPUDirect module; firmware.
- **NVLink:** link count and NVSwitch fabric state; NVSwitch count; Fabric Manager status and version.
- **RDMA:** port state and rate; the GPU-to-NIC rail map; `ib_write_bw` loopback through the switch on every rail.
- **DCGM:** `dcgmi diag -r 3`, plus per-GPU PCIe bandwidth floors.
- **NCCL:** single-node `all_reduce_perf`, checking bus bandwidth, correctness and NCCL version.
- **fio:** optional local-storage floors.
- **Health deltas:** XIDs, ECC, remapped rows, and IB and NVLink error counters, compared before and after the load.

The runner installs the tools it needs (DCGM, perftest, fio, a pinned nccl-tests build). It never touches the GPU driver, Fabric Manager, OFED/DOCA or the kernel. A run takes about 30–40 minutes. Most of that is DCGM level 3. The first run adds about 5–10 minutes for installs.

```sh
scripts/build-dist.sh                                   # -> dist/gfv-accept.sh, one self-contained file
scp dist/gfv-accept.sh node:
ssh node sudo bash gfv-accept.sh accept --class hgx-h100-8x
```

The same file can be served from anywhere and piped into bash:

```sh
curl -fsSL https://<host>/gfv-accept.sh | sudo bash -s -- accept --class hgx-h100-8x
```

**Exit codes:**
- `0`: the node passed.
- `1`: a check failed or could not run.
- `2`: the gate itself could not run. Causes: not root, a bad manifest, no working driver, or busy GPUs without `--force`.

**Outputs:**
- **stdout:** one JSON result line per check.
- **Evidence directory:** `/var/lib/gpu-fleet-validation/runs/<run_id>/`, holding `SUMMARY.md`, `results.jsonl`, `junit.xml` and every raw tool output.

See [docs/acceptance.md](docs/acceptance.md) for the checks, the manifest keys and the runbook.

## Status of the L2 gate

The parsers and the whole runner are tested against recorded output from a real 8x H100 HGX node with NDR InfiniBand. The tests use fake tools and a fake `/sys` tree; see [tests/fixtures/README.md](tests/fixtures/README.md). The installer is tested end to end in an Ubuntu 24.04 container. The runner has not yet run on GPU hardware. [docs/acceptance.md](docs/acceptance.md#first-hardware-run) lists what that first run must confirm.

**Known finding:** on the reference node, DCGM measured 76 GB/s bidirectional PCIe bandwidth on GPU 0 and GPU 1, against 101 GB/s on GPUs 2–7. The shipped floor is 95 GB/s, so replaying that node fails exactly those two checks. The tests assert that result.

## Development

```sh
python3 -m unittest discover -s tests -v     # Python 3.10+; nothing to install
scripts/build-dist.sh                        # needs Python 3.11+ (TOML manifests -> bundled JSON)
bash dist/gfv-accept.sh accept --class hgx-h100-8x --dry-run
```

The runtime uses only the Python standard library. The node needs nothing but `python3` ≥ 3.10, and the single file installs that on Ubuntu if it is missing.

## License

MIT
