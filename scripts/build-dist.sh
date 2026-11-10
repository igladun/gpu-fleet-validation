#!/usr/bin/env bash
# Build dist/gfv-accept.sh: the package, the bundled manifests (TOML converted to JSON) and the tool
# pins, zipped into a Python zipapp and embedded as base64 in a bash wrapper. One text file, no deps.
#
#   scripts/build-dist.sh                 # writes dist/gfv-accept.sh
#   DIST_DIR=/tmp/x scripts/build-dist.sh
set -euo pipefail

cd "$(dirname "$0")/.."
DIST_DIR="${DIST_DIR:-dist}"
python3 -c 'import sys, tomllib' 2>/dev/null || { echo "build needs Python >= 3.11 (tomllib)" >&2; exit 2; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/src" "$DIST_DIR"
cp -R gfv "$stage/src/gfv"
find "$stage/src" -name __pycache__ -type d -prune -exec rm -rf {} +

commit=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain -- gfv manifests 2>/dev/null)" ]; then commit="$commit-dirty"; fi
built=$(date -u +%Y-%m-%dT%H:%M:%SZ)
version=$(python3 -c 'import gfv; print(gfv.__version__)')

python3 - "$stage/src/gfv" "$commit" "$built" <<'PY'
import json, os, sys, tomllib
pkg, commit, built = sys.argv[1:4]
os.makedirs(os.path.join(pkg, "manifests"))
names = []
for fname in sorted(os.listdir("manifests")):
    if fname.endswith(".toml"):
        with open(os.path.join("manifests", fname), "rb") as f:
            doc = tomllib.load(f)
        out = fname[:-5] + ".json"
        with open(os.path.join(pkg, "manifests", out), "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        names.append(out)
with open(os.path.join(pkg, "manifests", "index.json"), "w") as f:
    json.dump(names, f)
with open(os.path.join(pkg, "_build.json"), "w") as f:
    json.dump({"commit": commit, "built": built}, f)
PY

# Own entry point: zipapp's generated one (-m) discards main()'s return value, i.e. the exit code.
printf 'import sys\nfrom gfv.__main__ import main\nsys.exit(main())\n' >"$stage/src/__main__.py"
python3 -m zipapp "$stage/src" -c -o "$stage/gfv.pyz"
python3 - "$stage/gfv.pyz" scripts/wrapper.sh.in "$DIST_DIR/gfv-accept.sh" "$version" "$commit" "$built" <<'PY'
import base64, sys
pyz, template, out, version, commit, built = sys.argv[1:7]
payload = base64.encodebytes(open(pyz, "rb").read()).decode().rstrip("\n")
text = open(template).read()
for k, v in {"@VERSION@": version, "@COMMIT@": commit, "@BUILT@": built, "@PAYLOAD@": payload}.items():
    text = text.replace(k, v)
with open(out, "w") as f:
    f.write(text)
PY
chmod +x "$DIST_DIR/gfv-accept.sh"
echo "built $DIST_DIR/gfv-accept.sh ($(wc -c <"$DIST_DIR/gfv-accept.sh") bytes, gfv $version, $commit)"
