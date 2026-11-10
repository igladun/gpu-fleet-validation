#!/usr/bin/env bash
# Build gfv-accept.sh: the package, the bundled manifests (TOML converted to JSON) and the tool pins in a
# Python zipapp, embedded as base64 in a bash wrapper. One text file, no dependencies.
#
# The build is reproducible: same sources -> same bytes on any OS (sorted entries, fixed timestamps,
# stored not deflated, a source hash instead of a commit or build time). The file is committed at the
# repo root so it can be curled from GitHub; tests/test_dist.py fails when it is stale.
#
#   scripts/build-dist.sh                 # writes ./gfv-accept.sh
#   DIST_DIR=/tmp/x scripts/build-dist.sh # writes /tmp/x/gfv-accept.sh
set -euo pipefail

cd "$(dirname "$0")/.."
DIST_DIR="${DIST_DIR:-.}"
python3 -c 'import sys, tomllib' 2>/dev/null || { echo "build needs Python >= 3.11 (tomllib)" >&2; exit 2; }
mkdir -p "$DIST_DIR"

python3 - "$DIST_DIR/gfv-accept.sh" <<'PY'
import base64, hashlib, io, json, os, sys, tomllib, zipfile

out = sys.argv[1]
EPOCH = (1980, 1, 1, 0, 0, 0)


def sources():
    """Every input of the build, as sorted (relative path, bytes)."""
    files = []
    for root, dirs, names in os.walk("gfv"):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        files += [os.path.join(root, n) for n in names if not n.endswith(".pyc")]
    files += [os.path.join("manifests", n) for n in os.listdir("manifests") if n.endswith(".toml")]
    files.append(os.path.join("scripts", "wrapper.sh.in"))
    return [(p.replace(os.sep, "/"), open(p, "rb").read()) for p in sorted(files)]


src = sources()
digest = hashlib.sha256()
for path, data in src:
    digest.update(path.encode() + b"\0" + hashlib.sha256(data).digest())
source = digest.hexdigest()[:12]

members = {path: data for path, data in src if path.startswith("gfv/")}
index = []
for path, data in src:
    if path.startswith("manifests/"):
        name = path[len("manifests/"):-len(".toml")] + ".json"
        members[f"gfv/manifests/{name}"] = (json.dumps(tomllib.loads(data.decode()), indent=1, sort_keys=True)
                                            + "\n").encode()
        index.append(name)
members["gfv/manifests/index.json"] = (json.dumps(sorted(index)) + "\n").encode()
members["gfv/_build.json"] = (json.dumps({"source": source}) + "\n").encode()
# Own entry point: zipapp's generated one discards main()'s return value, i.e. the exit code.
members["__main__.py"] = b"import sys\nfrom gfv.__main__ import main\nsys.exit(main())\n"

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
    for path in sorted(members):
        info = zipfile.ZipInfo(path, date_time=EPOCH)
        info.external_attr = 0o644 << 16
        info.create_system = 3                      # unix, regardless of the building OS
        z.writestr(info, members[path])

ns = {}
exec(compile(open("gfv/__init__.py").read(), "gfv/__init__.py", "exec"), ns)
version = ns["__version__"]
payload = base64.encodebytes(buf.getvalue()).decode().rstrip("\n")
text = open("scripts/wrapper.sh.in").read()
for k, v in {"@VERSION@": version, "@SOURCE@": source, "@PAYLOAD@": payload}.items():
    text = text.replace(k, v)
with open(out, "w", newline="\n") as f:
    f.write(text)
os.chmod(out, 0o755)
print(f"built {out} ({len(text.encode())} bytes, gfv {version}, src {source})")
PY
