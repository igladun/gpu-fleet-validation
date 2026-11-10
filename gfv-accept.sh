#!/usr/bin/env bash
{
# Retired: gfv-accept.sh used to hold the whole tool. It now ships as release assets; see README.md.
echo "gfv: gfv-accept.sh is retired; nothing was run. Use the release launcher instead:" >&2
echo "  curl -fsSL https://github.com/igladun/gpu-fleet-validation/releases/latest/download/gfv.sh | sudo bash -s -- accept --class hgx-h100-8x" >&2
exit 2
}
