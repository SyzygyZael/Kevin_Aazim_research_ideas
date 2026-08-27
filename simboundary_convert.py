#!/usr/bin/env python3
"""SimBoundary: convert a benchmark release into the interchange format.

Everything downstream of this file already works. This is the one step that
cannot be written without the benchmark in hand, because it depends on field
names and units that must be confirmed against the release. So the pipeline
here is complete and tested, and exactly one function is left for you:

    parse_source_record(rec) -> {"name", "devices", "properties"}

Write it for your benchmark, then run this file. It will convert, validate every
record with the same validator the loader uses, write the JSON Lines file, load
it back, and report per-rung class coverage so you can see immediately whether
the conversion produced enough repeated R-classes for the partition estimator to
have anything to average.

    python3 simboundary_convert.py --demo                    # worked example
    python3 simboundary_convert.py --in raw.jsonl --parser falcon --dry-run 5
    python3 simboundary_convert.py --in raw.jsonl --parser falcon --out falcon.jsonl

What "enough collisions" means: the partition estimator averages the spread of
the property within an R-equivalence class, so a class seen once contributes
nothing. If coverage at a rung is near zero the floor there is not measurable by
that estimator, which is a fact about the corpus and the rung, not a bug. Report
coverage rather than working around it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from simboundary_analysis import partition_floor
from simboundary_data import (FALCON_FIELDS_TO_CONFIRM, OCB_FIELDS_TO_CONFIRM,
                              SchemaError, _record_to_netlist, load_jsonl)
from simboundary_ladder import RUNGS, ladder_keys


# --------------------------------------------------------------------------
# the parsers
# --------------------------------------------------------------------------
def parse_example(rec: dict) -> dict:
    """Worked example for a toy source schema, used by --demo and the self-test.

    Source records look like:
        {"id": "x1",
         "elements": [["nmos", 1.4e-6, "in", "out", "gnd"], ["res", 1e4, "vdd", "out"]],
         "meas": {"gain": 22.5, "power": 1.2e-3}}

    Three-terminal elements are (kind, value, gate, drain, source); two-terminal
    elements are (kind, value, plus, minus). That mapping is the whole job.
    """
    devices = []
    for el in rec["elements"]:
        kind, value = str(el[0]), float(el[1])
        nets = [str(x) for x in el[2:]]
        if len(nets) == 3:
            terminals = {"g": nets[0], "d": nets[1], "s": nets[2]}
        elif len(nets) == 2:
            terminals = {"p": nets[0], "n": nets[1]}
        else:
            raise SchemaError(f"element {el!r} has {len(nets)} nets; expected 2 or 3")
        devices.append({"kind": kind, "value": value, "terminals": terminals})
    return {"name": str(rec.get("id", "")), "devices": devices,
            "properties": {str(k): float(v) for k, v in rec["meas"].items()}}


def _template(which: str, confirm: Dict[str, str]) -> Callable[[dict], dict]:
    def parser(rec: dict) -> dict:
        if (which == "FALCON"):
            pass

    return parser


parse_falcon = _template("FALCON", FALCON_FIELDS_TO_CONFIRM)
parse_ocb = _template("OCB", OCB_FIELDS_TO_CONFIRM)

PARSERS: Dict[str, Callable[[dict], dict]] = {
    "example": parse_example, "falcon": parse_falcon, "ocb": parse_ocb,
}


# --------------------------------------------------------------------------
# conversion pipeline (already complete)
# --------------------------------------------------------------------------
def read_source(path) -> Iterable[dict]:
    """Read JSON Lines, or a JSON array, or a JSON object of records."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"source file not found: {p}")
    text = p.read_text().strip()
    if not text:
        raise SchemaError(f"{p} is empty")
    if text[0] == "[":
        return list(json.loads(text))
    if text[0] == "{" and "\n" not in text.strip():
        obj = json.loads(text)
        return list(obj.values()) if isinstance(obj, dict) else [obj]
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError(f"source line {i}: not valid JSON ({e})") from None
    return out


def convert(source, out_path, parser: Callable[[dict], dict],
            limit: Optional[int] = None) -> dict:
    """Parse, validate, write. Returns a summary; raises on the first bad record."""
    records = list(read_source(source))
    if limit:
        records = records[:limit]
    written, prop_names = 0, None
    out_path = Path(out_path)
    with out_path.open("w") as fh:
        for i, rec in enumerate(records, 1):
            try:
                conv = parser(rec)
            except NotImplementedError:
                raise
            except Exception as e:
                raise SchemaError(f"source record {i}: parser failed ({e})") from None
            # validate with the same code the loader uses, so a record that
            # passes here cannot fail later
            _record_to_netlist(conv, i)
            if not conv.get("properties"):
                raise SchemaError(f"source record {i}: no properties produced")
            names = sorted(conv["properties"])
            if prop_names is None:
                prop_names = names
            elif names != prop_names:
                raise SchemaError(
                    f"source record {i}: properties {names} differ from {prop_names}. "
                    "Every record must carry the same properties, or the estimator "
                    "would silently average over different subsets.")
            fh.write(json.dumps(conv) + "\n")
            written += 1
    return {"records_in": len(records), "records_written": written,
            "properties": prop_names, "out": str(out_path)}


def report_coverage(jsonl_path, bin_width: float = 0.25) -> dict:
    """Load the converted corpus and report per-rung class coverage.

    Coverage is the fraction of circuits sitting in a class seen at least twice.
    It is the number that decides whether the partition estimator can say
    anything at a rung, so check it before running the full protocol.
    """
    corpus = load_jsonl(jsonl_path)
    keys = ladder_keys(corpus.netlists, bin_width=bin_width)
    prop = corpus.property_names()[0]
    y = corpus.properties[prop]
    rows = {}
    for r in RUNGS:
        est = partition_floor(keys[r], y)
        rows[r] = {"coverage": est["coverage"], "n_classes": est["n_classes"],
                   "distinct_keys": len(set(keys[r]))}
    return {"n_circuits": len(corpus), "properties": corpus.property_names(),
            "coverage_by_rung": rows, "stamp": corpus.stamp()}


def _print_report(summary: dict, cov: dict) -> None:
    print(f"\nwrote {summary['out']}: {summary['records_written']} circuits, "
          f"properties {summary['properties']}")
    print(f"corpus stamp: {cov['stamp']}")
    print("\nper-rung class coverage (fraction of circuits in a class seen >= 2 times):")
    for r, v in cov["coverage_by_rung"].items():
        bar = "#" * int(round(v["coverage"] * 40))
        print(f"  {r}  {v['coverage']:5.2f}  {bar:<40} "
              f"{v['n_classes']:>5} usable classes, {v['distinct_keys']:>5} distinct")
    low = [r for r, v in cov["coverage_by_rung"].items() if v["coverage"] < 0.05]
    if low:
        print(f"\nCoverage is near zero at {', '.join(low)}. The partition estimator "
              f"cannot\nmeasure the floor there. That is a property of the corpus, not a "
              f"bug: report\nthe coverage, lean on the neighborhood cross-check, and do "
              f"not quietly widen\nthe quantization bin to manufacture collisions.")


# ==========================================================================
# demo / self-test
# ==========================================================================
def run_demo(tmpdir: Optional[str] = None) -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        d = Path(tmpdir or d)
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        src = d / "raw.jsonl"
        # a toy source file with repeated topologies, so classes recur
        topos = [[["nmos", 1e-6, "in", "out", "gnd"], ["res", 1e4, "vdd", "out"]],
                 [["nmos", 2e-6, "in", "out", "gnd"], ["cap", 1e-12, "out", "gnd"]],
                 [["pmos", 4e-6, "in", "out", "vdd"], ["res", 2e4, "out", "gnd"]]]
        with src.open("w") as fh:
            for i in range(60):
                t = topos[i % len(topos)]
                fh.write(json.dumps({"id": f"x{i}", "elements": t,
                                     "meas": {"gain": 10.0 + (i % 3),
                                              "power": 1e-3 * (1 + (i % 3))}}) + "\n")
        out = d / "converted.jsonl"
        summary = convert(src, out, parse_example)
        cov = report_coverage(out)
        _print_report(summary, cov)

        ok &= summary["records_written"] == 60
        print(f"\n  [{'ok' if ok else 'FAIL'}] converted all 60 records")
        got = cov["coverage_by_rung"]["R2"]["coverage"] > 0.9
        print(f"  [{'ok' if got else 'FAIL'}] repeated topologies give high R2 coverage")
        ok &= got
        # a record with a bad element must be rejected, not silently dropped
        bad = d / "bad.jsonl"
        bad.write_text(json.dumps({"id": "b", "elements": [["nmos", 1.0, "a"]],
                                   "meas": {"gain": 1.0}}) + "\n")
        try:
            convert(bad, d / "x.jsonl", parse_example); raised = False
        except SchemaError:
            raised = True
        print(f"  [{'ok' if raised else 'FAIL'}] malformed source record raises")
        ok &= raised
        # the unwritten templates must refuse loudly
        try:
            parse_falcon({}); raised = False
        except NotImplementedError:
            raised = True
        print(f"  [{'ok' if raised else 'FAIL'}] unwritten FALCON parser refuses with guidance")
        ok &= raised
    print("\nDEMO:", "PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", help="source file from the benchmark")
    ap.add_argument("--out", help="destination .jsonl in the interchange format")
    ap.add_argument("--parser", choices=sorted(PARSERS), default="example")
    ap.add_argument("--dry-run", type=int, metavar="N",
                    help="parse the first N records, print them, write nothing")
    ap.add_argument("--demo", action="store_true", help="worked example end to end")
    a = ap.parse_args(argv)

    if a.demo:
        return 0 if run_demo() else 1
    if not a.src:
        ap.error("--in is required (or use --demo)")
    parser = PARSERS[a.parser]
    if a.dry_run:
        for i, rec in enumerate(list(read_source(a.src))[:a.dry_run], 1):
            conv = parser(rec)
            _record_to_netlist(conv, i)
            print(f"--- record {i} ---")
            print(json.dumps(conv, indent=2)[:900])
        print(f"\n{a.dry_run} records parsed and validated. Nothing written.")
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run is used")
    summary = convert(a.src, a.out, parser)
    _print_report(summary, report_coverage(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
