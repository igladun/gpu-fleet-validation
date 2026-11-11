# gpu-fleet-validation

Scripts that validate multi-node GPU training clusters: 8-GPU HGX-class nodes with one RDMA NIC per GPU, on an InfiniBand or RoCE fabric.

## Node-class manifest

Every check compares the node with a manifest. A manifest is a file that says what a healthy node of one class looks like: GPU count and model, driver and CUDA, NVLink and NVSwitches, RDMA rails, thermal limits and performance floors. Nothing in the code is specific to one class. The shipped class is [`hgx-h100-8x`](manifests/hgx-h100-8x.toml). For your own hardware, see [docs/manifest.md](docs/manifest.md).

## Run

Run it on the GPU node as root, with GPU workloads stopped. The node needs Ubuntu 22.04, 24.04 or 26.04 and a working NVIDIA driver. The script installs everything else and takes about an hour:

```sh
curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | sudo bash -s -- accept --class hgx-h100-8x
```

The verdict and evidence land in `/var/lib/gpu-fleet-validation/runs/<run_id>/SUMMARY.md`. Don't script on this pipe's exit code: a failed download also exits 0. For a dry run, runs over ssh, offline nodes, automation and every option, see [docs/acceptance.md](docs/acceptance.md#running-it).

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

## Development

See [docs/development.md](docs/development.md).

## License

MIT
