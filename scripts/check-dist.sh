#!/usr/bin/env bash
# Smoke-test built release assets: the launcher (from a file and from stdin) and gfv.pyz on its own, with
# every python3 >= 3.10 found on PATH (python3, python3.10, python3.12, ...). Used by CI and the release.
#
#   scripts/check-dist.sh [DIST_DIR]                          # default: dist
#   CHECK_PYTHONS="3.10 3.12" scripts/check-dist.sh dist      # also fail unless these versions are found
set -euo pipefail

dist=$(cd "${1:-dist}" && pwd)
shellcheck "$dist/gfv.sh"
(cd "$dist" && python3 -I -c 'import hashlib, sys
for line in open("SHA256SUMS"):
    want, name = line.split()
    got = hashlib.sha256(open(name, "rb").read()).hexdigest()
    sys.exit(f"{name}: sha256 {got} != {want}") if got != want else print(f"{name}: OK")')

pythons=()
for py in python3 python3.10 python3.11 python3.12 python3.13 python3.14; do
  p=$(command -v "$py" 2>/dev/null) || continue
  "$p" -I -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || continue
  real=$("$p" -I -c 'import os, sys; print(os.path.realpath(sys.executable))')
  case " ${pythons[*]-} " in *" $real "*) ;; *) pythons+=("$real") ;; esac
done
[ "${#pythons[@]}" -gt 0 ] || { echo "no python3 >= 3.10 on PATH" >&2; exit 2; }
for want in ${CHECK_PYTHONS:-}; do
  command -v "python$want" >/dev/null 2>&1 || { echo "check-dist: python$want required but not on PATH" >&2; exit 2; }
done

for py in "${pythons[@]}"; do
  echo "== $("$py" -V)"
  bin=$(mktemp -d)
  ln -s "$py" "$bin/python3"
  (
    export PATH="$bin:$PATH" GFV_PYZ="$dist/gfv.pyz"
    python3 -I "$dist/gfv.pyz" version
    python3 -I "$dist/gfv.pyz" classes
    bash "$dist/gfv.sh" version
    bash "$dist/gfv.sh" accept --class hgx-h100-8x --dry-run >/dev/null
    bash -s -- accept --class hgx-h100-8x --dry-run <"$dist/gfv.sh" >/dev/null
  )
  rm -rf "$bin"
done
echo "check-dist: OK"
