# Development

Run these from the repo root:

```sh
python3 -m unittest discover -s tests -v     # Python 3.10+; nothing to install
scripts/build-dist.sh                        # writes dist/gfv.pyz, dist/gfv.sh, dist/SHA256SUMS; needs Python 3.11+
scripts/check-dist.sh dist                   # smoke-tests them with every python3 >= 3.10 on PATH
GFV_PYZ=dist/gfv.pyz bash dist/gfv.sh accept --class hgx-h100-8x --dry-run
python3 -m gfv accept --class hgx-h100-8x --dry-run   # straight from the checkout
```

`gfv.sh` is rendered from [`scripts/bootstrap.sh.in`](../scripts/bootstrap.sh.in). `GFV_PYZ` makes it use a local `gfv.pyz` instead of downloading one; the sha256 check still applies. The build is reproducible: the same sources give the same bytes on any OS. Nothing built is committed.

The runtime uses only the Python standard library. The node needs nothing but `python3` ≥ 3.10, and `gfv.sh` installs that on Ubuntu if it is missing.

What the tests cover: [acceptance.md](acceptance.md#status).

## Releasing

Set `__version__` in [`gfv/__init__.py`](../gfv/__init__.py), commit to `main`, then push a tag `v<version>`.

The [release workflow](../.github/workflows/release.yml) then runs two jobs:
1. A read-only job tests and builds the files.
2. A second job attests the files, uploads them to a draft release and checks the draft against the build. It then publishes the release and runs the published one-liner.

A version with a suffix (`v0.2.0-rc1`) becomes a prerelease and never `latest`.
