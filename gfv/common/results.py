"""One result schema for every check: JSONL lines, JUnit XML and a Markdown summary."""

import datetime
import json
import os
import sys
from xml.sax.saxutils import escape, quoteattr

VERDICTS = ("pass", "fail", "warn", "info", "skip", "error")
FAILING = ("fail", "error")


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Recorder:
    """Collects results. Each result is appended to results.partial.jsonl at once (survives a crash);
    the final results.jsonl, junit.xml and SUMMARY.md are written by finish()."""

    def __init__(self, context, out_dir, log=sys.stderr):
        self.context = dict(context)
        self.versions = {}
        self.out_dir = out_dir
        self.results = []
        self.log = log
        self._partial = os.path.join(out_dir, "results.partial.jsonl")

    def add(self, check, verdict, target=None, value=None, expected=None, detail="", evidence=None):
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}")
        line = {
            "ts": utcnow(), **self.context, "check": check, "target": target, "value": value,
            "expected": expected, "verdict": verdict, "detail": detail, "versions": None,
            "evidence": evidence,
        }
        self.results.append(line)
        with open(self._partial, "a") as f:
            f.write(json.dumps({**line, "versions": self.versions}) + "\n")
        tgt = f" [{target}]" if target is not None else ""
        shown = "" if value is None else f" value={value}"
        exp = "" if expected is None else f" expected={expected}"
        msg = f" - {detail}" if detail else ""
        print(f"  {verdict.upper():5} {check}{tgt}{shown}{exp}{msg}", file=self.log, flush=True)
        return line

    def failing(self):
        return [r for r in self.results if r["verdict"] in FAILING]

    def overall(self):
        return "fail" if self.failing() else "pass"

    def counts(self):
        out = {v: 0 for v in VERDICTS}
        for r in self.results:
            out[r["verdict"]] += 1
        return out

    def finish(self, meta, stream=sys.stdout):
        """Write the final files (every line carries the final versions) and print JSONL to stream."""
        lines = [{**r, "versions": self.versions} for r in self.results]
        with open(os.path.join(self.out_dir, "results.jsonl"), "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        if stream is not None:
            for line in lines:
                stream.write(json.dumps(line) + "\n")
            stream.flush()
        with open(os.path.join(self.out_dir, "versions.json"), "w") as f:
            json.dump(self.versions, f, indent=2, sort_keys=True)
        with open(os.path.join(self.out_dir, "junit.xml"), "w") as f:
            f.write(self.junit())
        with open(os.path.join(self.out_dir, "SUMMARY.md"), "w") as f:
            f.write(self.summary(meta))

    def junit(self):
        c = self.counts()
        suite = (f'<testsuite name="gfv-{self.context.get("tier", "L2")}" tests="{len(self.results)}" '
                 f'failures="{c["fail"]}" errors="{c["error"]}" skipped="{c["skip"]}">')
        out = ['<?xml version="1.0" encoding="UTF-8"?>', "<testsuites>", suite]
        for r in self.results:
            name = r["check"] + (f"[{r['target']}]" if r["target"] is not None else "")
            stage = r["check"].split(".", 1)[0]
            body = ""
            text = f"value={r['value']} expected={r['expected']} {r['detail']}".strip()
            if r["verdict"] == "fail":
                body = f"<failure message={quoteattr(text)}/>"
            elif r["verdict"] == "error":
                body = f"<error message={quoteattr(text)}/>"
            elif r["verdict"] == "skip":
                body = f"<skipped message={quoteattr(r['detail'])}/>"
            elif r["verdict"] in ("warn", "info"):
                body = f"<system-out>{escape(r['verdict'] + ': ' + text)}</system-out>"
            out.append(f'<testcase classname={quoteattr("L2." + stage)} name={quoteattr(name)}>{body}</testcase>')
        out += ["</testsuite>", "</testsuites>", ""]
        return "\n".join(out)

    def summary(self, meta):
        c = self.counts()
        verdict = self.overall().upper()
        ctx = self.context
        lines = [
            f"# L2 acceptance: {ctx.get('node')}",
            "",
            f"**Verdict: {verdict}** · class `{ctx.get('class')}` · run `{ctx.get('run_id')}` · "
            f"{meta.get('finished', utcnow())}",
            "",
            "Counts: " + ", ".join(f"{k} {v}" for k, v in c.items() if v),
            "",
            "Versions: " + (", ".join(f"{k} `{v}`" for k, v in sorted(self.versions.items())) or "-"),
            "",
        ]
        for note in meta.get("notes", []):
            lines.append(f"> {note}")
        if meta.get("notes"):
            lines.append("")
        order = {"fail": 0, "error": 1, "warn": 2, "skip": 3, "info": 4, "pass": 5}
        rows = sorted(self.results, key=lambda r: order[r["verdict"]])
        lines += ["| Verdict | Check | Target | Value | Expected | Detail |", "|---|---|---|---|---|---|"]
        for r in rows:
            cells = [r["verdict"].upper(), r["check"], r["target"], r["value"], r["expected"], r["detail"]]
            lines.append("| " + " | ".join(_md(x) for x in cells) + " |")
        lines.append("")
        return "\n".join(lines)


def _md(x):
    if x is None:
        return ""
    s = json.dumps(x) if isinstance(x, (list, dict)) else str(x)
    return s.replace("|", "\\|").replace("\n", " ")
