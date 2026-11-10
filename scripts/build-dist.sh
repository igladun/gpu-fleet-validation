#!/usr/bin/env bash
# Build the release assets:
#   gfv.pyz     the package, the bundled manifests (TOML converted to JSON) and the tool pins, as a Python zipapp
#   gfv.sh      the launcher (scripts/bootstrap.sh.in) with the version and the sha256 of gfv.pyz baked in
#   SHA256SUMS  both files
#
# The build is reproducible: same sources -> same bytes on any OS (sorted entries, fixed timestamps,
# stored not deflated, a source hash instead of a commit or build time). The release workflow builds
# these from a version tag and publishes them; nothing built is committed.
#
#   scripts/build-dist.sh                 # writes dist/ in the repo
#   DIST_DIR=/tmp/x scripts/build-dist.sh # writes /tmp/x/
#   scripts/check-dist.sh dist            # smoke-tests the result
set -euo pipefail

# A relative DIST_DIR is relative to the caller's directory; the default is the repo's dist/.
case "${DIST_DIR:-}" in "" | /*) ;; *) DIST_DIR="$PWD/$DIST_DIR" ;; esac
cd "$(dirname "$0")/.."
DIST_DIR="${DIST_DIR:-dist}"
python3 -c 'import sys, tomllib' 2>/dev/null || { echo "build needs Python >= 3.11 (tomllib)" >&2; exit 2; }
mkdir -p "$DIST_DIR"

python3 - "$DIST_DIR" <<'PY'
import hashlib, io, json, os, re, sys, tomllib, zipfile

out_dir = sys.argv[1]
EPOCH = (1980, 1, 1, 0, 0, 0)


def sources():
    """Every input of gfv.pyz, as sorted (relative path, bytes). The launcher template is not one.
    gfv/ may hold only .py and .json files: editor and OS litter is skipped, anything else stops the build
    (a data file silently left out would only fail later, on the node)."""
    files = []
    for root, dirs, names in os.walk("gfv"):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for n in names:
            if n.endswith((".py", ".json")):
                files.append(os.path.join(root, n))
            elif not (n in (".DS_Store",) or n.endswith((".pyc", ".swp", "~"))):
                sys.exit(f"build: unexpected file {os.path.join(root, n)}; gfv/ ships only .py and .json")
    files += [os.path.join("manifests", n) for n in os.listdir("manifests") if n.endswith(".toml")]
    return [(p.replace(os.sep, "/"), open(p, "rb").read()) for p in sorted(files)]


def write(name, data, mode):
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    os.chmod(path, mode)
    return hashlib.sha256(data).hexdigest()


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
if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", version):
    sys.exit(f"build: __version__ {version!r} is not X.Y.Z or X.Y.Z-suffix")

pyz_sha = write("gfv.pyz", buf.getvalue(), 0o644)
text = open("scripts/bootstrap.sh.in").read()
for k, v in {"@VERSION@": version, "@SHA256@": pyz_sha}.items():
    if k not in text:
        sys.exit(f"build: {k} missing from scripts/bootstrap.sh.in")
    text = text.replace(k, v)
left = re.findall(r"@[A-Z0-9_]+@", text)
if left:
    sys.exit(f"build: unreplaced {sorted(set(left))} in scripts/bootstrap.sh.in")
sh_sha = write("gfv.sh", text.encode(), 0o755)
write("SHA256SUMS", f"{pyz_sha}  gfv.pyz\n{sh_sha}  gfv.sh\n".encode(), 0o644)
print(f"built {out_dir}/gfv.pyz ({len(buf.getvalue())} bytes, gfv {version}, src {source}, sha256 {pyz_sha})")
print(f"built {out_dir}/gfv.sh ({len(text.encode())} bytes)")
PY
