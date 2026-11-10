"""Optional local-storage check: sequential write then read with fio, against per-class floors."""

import os

from . import parse


def run(ctx):
    st = ctx.table("storage")
    if not st:
        ctx.add("fio", "skip", None, None, None, "no [storage] table in the manifest")
        return
    if not ctx.which("fio"):
        ctx.add("fio", "error", None, None, None, "fio not found")
        return
    path = st["path"]
    if not os.path.isdir(path):
        ctx.add("fio", "error", path, None, None, "storage.path is not a directory")
        return
    size = int(st.get("size_gib", 16))
    ctx.stage("fio", f"sequential write + read, {size} GiB in {path}")
    data = os.path.join(path, f"gfv-fio-{ctx.rec.context['run_id']}.dat")
    try:
        for rw, key in (("write", "seq_write_gbs"), ("read", "seq_read_gbs")):
            r = ctx.run(f"fio-seq{rw}", ["fio", f"--name=gfv-seq{rw}", f"--filename={data}", f"--rw={rw}",
                                         "--bs=1M", f"--size={size}G", "--direct=1", "--ioengine=libaio",
                                         "--iodepth=32", "--numjobs=1", "--group_reporting",
                                         "--output-format=json"], timeout=1800)
            try:
                gbs = round(parse.fio_gbs(r.out, rw), 3)
            except (ValueError, KeyError, IndexError) as e:
                ctx.add(f"fio.seq_{rw}", "error", path, None, st.get(key), f"{r.why()}; {e}", r.evidence)
                continue
            floor = st.get(key)
            if floor is None:
                ctx.add(f"fio.seq_{rw}", "info", path, gbs, None, "GB/s; no floor in manifest", r.evidence)
            else:
                ctx.add(f"fio.seq_{rw}", "pass" if gbs >= floor else "fail", path, gbs, floor, "GB/s", r.evidence)
    finally:
        try:
            os.remove(data)
        except OSError:
            pass
