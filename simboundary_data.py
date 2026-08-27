#!/usr/bin/env python3
"""SimBoundary: corpus loading.

The benchmarks ship in their own formats, and their exact field names are a
standing blocker (confirm against the release before trusting any number). So
this module defines one small interchange format, validates it strictly, and
refuses to invent data when it is missing.

Interchange format: JSON Lines, one circuit per line.

    {"name": "ckt_00017",
     "devices": [{"kind": "nmos", "value": 1.4e-6,
                  "terminals": {"g": "in", "d": "out", "s": "gnd"}},
                 {"kind": "res",  "value": 10000.0,
                  "terminals": {"p": "vdd", "n": "out"}}],
     "properties": {"gain_db": 24.1, "bandwidth_hz": 3.2e8, "dc_power_w": 1.1e-3}}

Rules this module enforces:

  * A missing or malformed file raises. There is no synthetic fallback. A
    pipeline that quietly substitutes fabricated circuits for real ones will
    produce numbers that look like measurements and are not.
  * Synthetic corpora exist for pipeline testing only, are produced by an
    explicitly named function, and carry `synthetic=True`. Every downstream
    artifact built from them is stamped SYNTHETIC.

Run `python3 simboundary_data.py` for the self-test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simboundary_ladder import Device, Netlist

REQUIRED_DEVICE_FIELDS = ("kind", "terminals")
REQUIRED_RECORD_FIELDS = ("devices", "properties")


class SchemaError(ValueError):
    """Raised when a record does not match the interchange format."""


@dataclass
class Corpus:
    netlists: List[Netlist]
    properties: Dict[str, np.ndarray]
    name: str = ""
    synthetic: bool = False
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.netlists)

    def property_names(self) -> List[str]:
        return sorted(self.properties)

    def subset(self, idx) -> "Corpus":
        idx = np.asarray(idx)
        return Corpus([self.netlists[i] for i in idx],
                      {k: v[idx] for k, v in self.properties.items()},
                      name=self.name, synthetic=self.synthetic, notes=list(self.notes))

    def stamp(self) -> str:
        return "SYNTHETIC (pipeline test only)" if self.synthetic else "real data"


def _record_to_netlist(rec: dict, line_no: int) -> Netlist:
    for f in REQUIRED_RECORD_FIELDS:
        if f not in rec:
            raise SchemaError(f"line {line_no}: record is missing required field {f!r}. "
                              f"Required: {REQUIRED_RECORD_FIELDS}")
    devices = []
    for k, d in enumerate(rec["devices"]):
        for f in REQUIRED_DEVICE_FIELDS:
            if f not in d:
                raise SchemaError(f"line {line_no}, device {k}: missing {f!r}. "
                                  f"Required: {REQUIRED_DEVICE_FIELDS}")
        if not isinstance(d["terminals"], dict) or not d["terminals"]:
            raise SchemaError(f"line {line_no}, device {k}: 'terminals' must be a "
                              f"non-empty {{role: net}} mapping")
        devices.append(Device(kind=str(d["kind"]),
                              terminals={str(r): str(n) for r, n in d["terminals"].items()},
                              value=(None if d.get("value") is None else float(d["value"])),
                              orient=d.get("orient")))
    if not devices:
        raise SchemaError(f"line {line_no}: record has no devices")
    return Netlist(devices, name=str(rec.get("name", f"ckt_{line_no}")))


def load_jsonl(path, name: str = "", require_properties: Optional[List[str]] = None) -> Corpus:
    """Load a corpus from the interchange format. Raises if anything is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"corpus file not found: {p}\n"
            "Convert the benchmark to the JSON Lines format documented at the top of "
            "simboundary_data.py first. There is deliberately no synthetic fallback: "
            "see simboundary_runbook.txt.")
    netlists, props, meta = [], {}, {}
    with p.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"line {i}: not valid JSON ({e})") from None
            if "_meta" in rec:                       # metadata line, not a circuit
                meta.update(rec["_meta"])
                continue
            netlists.append(_record_to_netlist(rec, i))
            for k, v in rec["properties"].items():
                props.setdefault(k, []).append(float(v) if v is not None else np.nan)
    if not netlists:
        raise SchemaError(f"{p} contained no records")
    n = len(netlists)
    for k, v in props.items():
        if len(v) != n:
            raise SchemaError(f"property {k!r} present on {len(v)} of {n} records; "
                              "properties must be present on every record or absent entirely")
    corpus = Corpus(netlists, {k: np.asarray(v, dtype=float) for k, v in props.items()},
                    name=name or p.stem, synthetic=bool(meta.get("synthetic", False)))
    if meta.get("note"):
        corpus.notes.append(str(meta["note"]))
    if require_properties:
        missing = [q for q in require_properties if q not in corpus.properties]
        if missing:
            raise SchemaError(f"{p} is missing required properties {missing}; "
                              f"present: {corpus.property_names()}")
    return corpus


def export_jsonl(corpus: Corpus, path) -> None:
    """Write a corpus in the interchange format, preserving the synthetic stamp.

    Without this marker the stamp is lost on a round trip through disk and a
    synthetic corpus would be indistinguishable from measured data downstream.
    """
    p = Path(path)
    with p.open("w") as fh:
        if corpus.synthetic or corpus.notes:
            fh.write(json.dumps({"_meta": {"synthetic": bool(corpus.synthetic),
                                           "note": "; ".join(corpus.notes)}}) + "\n")
        for i, nl in enumerate(corpus.netlists):
            fh.write(json.dumps({
                "name": nl.name,
                "devices": [{"kind": d.kind, "value": d.value, "terminals": d.terminals}
                            for d in nl.devices],
                "properties": {k: float(v[i]) for k, v in corpus.properties.items()}}) + "\n")


# --------------------------------------------------------------------------
# benchmark adapters
# --------------------------------------------------------------------------
FALCON_FIELDS_TO_CONFIRM = {
    "device kind": "CONFIRM against the FALCON release",
    "device sizing value": "CONFIRM (W/L units)",
    "terminal roles": "CONFIRM (naming of gate/drain/source)",
    "noise figure": "CONFIRM (dB or linear)",
    "dc power": "CONFIRM (W or mW)",
}
OCB_FIELDS_TO_CONFIRM = {
    "subgraph/device encoding": "CONFIRM against Ckt-Bench-101/301",
    "gain / bandwidth / phase margin / FoM": "CONFIRM units and sign conventions",
}


def load_falcon(jsonl_path, require=("dc_power_w",)) -> Corpus:
    """FALCON, after conversion to the interchange format.

    The converter is the student's to write and verify: field names and units
    must be confirmed against the release, which is an open blocker recorded in
    the runbook. This loader only checks that the converted file is well formed.
    """
    c = load_jsonl(jsonl_path, name="FALCON", require_properties=list(require))
    c.notes.append("field names/units CONFIRM against FALCON release: "
                   + "; ".join(f"{k} [{v}]" for k, v in FALCON_FIELDS_TO_CONFIRM.items()))
    return c


def load_ocb(jsonl_path, require=("gain_db",)) -> Corpus:
    """Open Circuit Benchmark, after conversion to the interchange format."""
    c = load_jsonl(jsonl_path, name="OCB", require_properties=list(require))
    c.notes.append("field names/units CONFIRM against OCB release: "
                   + "; ".join(f"{k} [{v}]" for k, v in OCB_FIELDS_TO_CONFIRM.items()))
    return c


# --------------------------------------------------------------------------
# synthetic corpus: pipeline testing ONLY, never a stand-in for measurement
# --------------------------------------------------------------------------
def make_synthetic_corpus(n: int = 400, seed: int = 0,
                          hidden_noise_sd: float = 0.0) -> Corpus:
    """A small corpus with a KNOWN generating rule, for exercising the pipeline.

    `p_graph` is a deterministic function of the sized netlist, so its floor at
    a rung that resolves device values must be ~0. `p_hidden` adds Gaussian
    noise of known variance that no graph rung carries, so its floor at every
    rung must be ~`hidden_noise_sd**2`. That is what makes this usable as a
    positive control for the estimator, and useless as evidence about circuits.
    """
    rng = np.random.default_rng(seed)
    kinds = ["nmos", "pmos", "res", "cap"]
    nets = ["vdd", "gnd", "n1", "n2"]

    # Draw a modest number of distinct templates and repeat them, so that
    # R-equivalence classes actually recur. Without repeated classes the
    # partition estimator has nothing to average over and returns nan, which
    # is the sparse-collision regime the paper reports coverage for.
    n_templates = max(8, n // 12)
    templates = []
    for _ in range(n_templates):
        devs, acc = [], 0.0
        for _ in range(int(rng.integers(2, 5))):
            kind = kinds[int(rng.integers(len(kinds)))]
            val = float(10 ** rng.uniform(0, 3))
            if kind in ("res", "cap"):
                t = {"p": nets[int(rng.integers(4))], "n": nets[int(rng.integers(4))]}
            else:
                t = {"g": nets[int(rng.integers(4))], "d": nets[int(rng.integers(4))],
                     "s": nets[int(rng.integers(4))]}
            devs.append(Device(kind, t, val))
            acc += np.log10(val) * (1.0 + kinds.index(kind))
        templates.append((devs, acc))

    netlists, p_graph = [], []
    for i in range(n):
        devs, acc = templates[int(rng.integers(n_templates))]
        netlists.append(Netlist([Device(dv.kind, dict(dv.terminals), dv.value)
                                 for dv in devs], name=f"synth_{i:05d}"))
        p_graph.append(acc)
    p_graph = np.asarray(p_graph)
    props = {"p_graph": p_graph,
             "p_hidden": p_graph + rng.normal(0, hidden_noise_sd, size=n)}
    return Corpus(netlists, props, name="synthetic", synthetic=True,
                  notes=[f"synthetic; hidden_noise_var={hidden_noise_sd**2:.6g}"])


# ==========================================================================
# SELF-TEST
# ==========================================================================
def run_self_test() -> bool:
    import tempfile
    ok = True

    c = make_synthetic_corpus(50, hidden_noise_sd=0.3)
    print(f"  [{'ok' if c.synthetic else 'FAIL'}] synthetic corpus is stamped: {c.stamp()}")
    ok &= c.synthetic

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.jsonl"
        with p.open("w") as fh:
            for nl, g in zip(c.netlists, c.properties["p_graph"]):
                fh.write(json.dumps({
                    "name": nl.name,
                    "devices": [{"kind": dv.kind, "value": dv.value,
                                 "terminals": dv.terminals} for dv in nl.devices],
                    "properties": {"p_graph": float(g)}}) + "\n")
        back = load_jsonl(p)
        same = len(back) == len(c) and np.allclose(back.properties["p_graph"],
                                                   c.properties["p_graph"])
        print(f"  [{'ok' if same else 'FAIL'}] round-trip through the interchange format")
        ok &= same
        print(f"  [{'ok' if not back.synthetic else 'FAIL'}] loaded corpus is not "
              f"flagged synthetic: {back.stamp()}")
        ok &= not back.synthetic

        p2 = Path(d) / "synth.jsonl"
        export_jsonl(c, p2)
        kept = load_jsonl(p2).synthetic
        print(f"  [{'ok' if kept else 'FAIL'}] synthetic stamp survives a round trip "
              f"through disk")
        ok &= kept

        try:
            load_jsonl(Path(d) / "missing.jsonl"); raised = False
        except FileNotFoundError:
            raised = True
        print(f"  [{'ok' if raised else 'FAIL'}] missing file raises instead of "
              f"falling back to synthetic")
        ok &= raised

        bad = Path(d) / "bad.jsonl"
        bad.write_text(json.dumps({"devices": [{"kind": "nmos"}],
                                   "properties": {"x": 1.0}}) + "\n")
        try:
            load_jsonl(bad); raised = False
        except SchemaError:
            raised = True
        print(f"  [{'ok' if raised else 'FAIL'}] malformed record raises SchemaError")
        ok &= raised

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    print("SimBoundary data self-test")
    sys.exit(0 if run_self_test() else 1)
