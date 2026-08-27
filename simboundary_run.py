#!/usr/bin/env python3
"""SimBoundary: run the experiments.

Implements the protocol of the paper's experimental-design section end to end
for everything computable without training a model, and computes the floor/gap
split for the two experiments that do need trained-model errors.

    E1  positive control (GATE: nothing else is reported unless E1 passes)
    E2  floor curves along the ladder, per property; minimum sufficient rung
    E4  collision case studies
    E5  cross-benchmark consistency
    E7  oracle validation: inject non-netlist variance, check the floor tracks it
    E8  estimator robustness: bin width, coverage, cross-estimator agreement
    E3/E6  floor-vs-gap decomposition, from a JSON of measured model errors

Usage
    python3 simboundary_run.py --self-test
    python3 simboundary_run.py --corpus falcon.jsonl --out results/
    python3 simboundary_run.py --corpus falcon.jsonl --corpus-b ocb.jsonl \\
        --model-errors errors.json --out results/

Results from a synthetic corpus are refused unless --allow-synthetic, and every
artifact built from one is stamped SYNTHETIC.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simboundary_analysis import (BASE_SEED, bootstrap_floor_ci, decomposition,
                                  neighborhood_floor, partition_floor,
                                  permutation_null)
from simboundary_data import (Corpus, SchemaError, load_jsonl,
                              make_synthetic_corpus)
from simboundary_ladder import RUNGS, build_graph, ladder_keys

# Pre-registered before any run (see the reproducibility appendix).
TOLERANCE_FRAC = 0.05      # "near zero" = 5% of the property's total variance
N_BOOT = 1000
N_PERM = 500
BIN_WIDTH = 0.25


def _floor_at(keys, y, n_boot=N_BOOT, seed=BASE_SEED):
    m = np.isfinite(y)
    keys, y = [k for k, ok in zip(keys, m) if ok], np.asarray(y)[m]
    est = partition_floor(keys, y)
    lo, hi = bootstrap_floor_ci(keys, y, n_boot=n_boot, seed=seed)
    est.update(ci_lo=lo, ci_hi=hi, n=int(len(y)))
    return est


def _featurize(netlist, rung: str) -> np.ndarray:
    """Deterministic fixed-length embedding, used only for the E8 cross-check."""
    adj, labels = build_graph(netlist, rung, bin_width=BIN_WIDTH)
    deg = adj.sum(1)
    counts = defaultdict(float)
    for lab, d in zip(labels, deg):
        counts[lab] += 1.0
        counts[lab + "#deg"] += float(d)
    keys = sorted(counts)
    v = np.array([counts[k] for k in keys], dtype=float)
    # hash label strings into a fixed 32-dim space so vectors are comparable
    out = np.zeros(32)
    for k, val in zip(keys, v):
        out[hash(k) % 32] += val
    return out


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def e1_positive_control(corpus: Corpus, keys, prop: str, resolving_rung: str = "R2") -> dict:
    """Gate. An easy property must show a near-zero floor once the rung resolves
    it, and the permutation null must return roughly the total variance."""
    y = corpus.properties[prop]
    est = _floor_at(keys[resolving_rung], y)
    tol = TOLERANCE_FRAC * est["total_var"]
    null = permutation_null(keys[resolving_rung], y, n_perm=N_PERM, seed=BASE_SEED)
    null_ratio = float(null["ratio_to_total_var"])
    passed = bool(np.isfinite(est["phi"]) and est["ci_hi"] < tol
                  and 0.8 < null_ratio < 1.2)
    if not np.isfinite(est["phi"]):
        why = (f"coverage={est['coverage']:.2f}: no R-class at {resolving_rung} was "
               "observed twice, so the partition estimator has nothing to average. "
               "Use a coarser rung, or widen the quantization bin and report that "
               "you did, or use a corpus with repeated topologies.")
    elif est["ci_hi"] >= tol:
        why = (f"floor CI upper bound {est['ci_hi']:.4g} is not below the tolerance "
               f"{tol:.4g}; this property is not graph-determined at {resolving_rung}, "
               "so it is the wrong positive control.")
    elif not 0.8 < null_ratio < 1.2:
        why = (f"permutation null returned {null_ratio:.2f} of total variance "
               "(expected ~1.0); the class labels and property values may be misaligned.")
    else:
        why = ""
    return {"property": prop, "rung": resolving_rung, "floor": est,
            "tolerance": tol, "null_ratio": null_ratio, "passed": passed,
            "diagnosis": why}


def e2_floor_curves(corpus: Corpus, keys) -> dict:
    """Main result: floor per rung per property, and the minimum sufficient rung."""
    out = {}
    for prop in corpus.property_names():
        y = corpus.properties[prop]
        curve, tol = {}, None
        for r in RUNGS:
            est = _floor_at(keys[r], y)
            if tol is None:
                tol = TOLERANCE_FRAC * est["total_var"]
            curve[r] = est
        msr = next((r for r in RUNGS if curve[r]["ci_hi"] < tol), None)
        nonincreasing = all(curve[a]["phi"] >= curve[b]["phi"] - 1e-9
                            for a, b in zip(RUNGS, RUNGS[1:]))
        out[prop] = {"curve": curve, "tolerance": tol,
                     "min_sufficient_rung": msr,
                     "verdict": "graph-determined" if msr else "no rung below tolerance",
                     "monotone_nonincreasing": nonincreasing}
    return out


def e4_collisions(corpus: Corpus, keys, rung: str = "R2", top: int = 5) -> dict:
    """Largest within-class spreads: the circuits a rung cannot tell apart."""
    out = {}
    for prop in corpus.property_names():
        y = corpus.properties[prop]
        groups = defaultdict(list)
        for i, k in enumerate(keys[rung]):
            if np.isfinite(y[i]):
                groups[k].append(i)
        cases = []
        for k, idx in groups.items():
            if len(idx) >= 2:
                vals = y[idx]
                cases.append({"key": k[:12], "n": len(idx),
                              "within_var": float(np.var(vals, ddof=1)),
                              "spread": float(vals.max() - vals.min()),
                              "members": [corpus.netlists[i].name for i in idx[:4]]})
        cases.sort(key=lambda c: -c["within_var"])
        out[prop] = cases[:top]
    return out


def e5_cross_benchmark(res_a: dict, res_b: dict, name_a: str, name_b: str) -> dict:
    shared = sorted(set(res_a) & set(res_b))
    return {"shared_properties": shared,
            "agreement": {p: {name_a: res_a[p]["min_sufficient_rung"],
                              name_b: res_b[p]["min_sufficient_rung"],
                              "agree": res_a[p]["min_sufficient_rung"] ==
                                       res_b[p]["min_sufficient_rung"]}
                          for p in shared}}


def e7_oracle_injection(corpus: Corpus, keys, base_prop: str,
                        sds=(0.0, 0.25, 0.5, 1.0), rung: str = "R4",
                        seed: int = BASE_SEED) -> dict:
    """Inject non-netlist variance of known size and check the floor tracks it.

    Proposition 1 says the floor at the finest rung equals the variance of the
    property from factors the netlist does not carry. Adding noise of known
    variance to a graph-determined label makes that quantity known, so the
    estimated floor should follow it with slope ~1.
    """
    rng = np.random.default_rng(seed)
    y0 = corpus.properties[base_prop]
    rows = []
    for sd in sds:
        y = y0 + rng.normal(0, sd, size=len(y0))
        est = _floor_at(keys[rung], y, n_boot=200)
        rows.append({"injected_var": float(sd ** 2), "estimated_floor": est["phi"],
                     "ci": [est["ci_lo"], est["ci_hi"]], "coverage": est["coverage"]})
    x = np.array([r["injected_var"] for r in rows])
    z = np.array([r["estimated_floor"] for r in rows])
    ok = np.isfinite(z).all() and len(set(x)) > 1
    slope, intercept = (np.polyfit(x, z, 1) if ok else (float("nan"),) * 2)
    return {"base_property": base_prop, "rung": rung, "rows": rows,
            "slope": float(slope), "intercept": float(intercept),
            "slope_near_one": bool(ok and abs(slope - 1.0) < 0.25)}


def e8_robustness(corpus: Corpus, prop: str, rung: str = "R2",
                  widths=(0.10, 0.25, 0.50, 1.00), kappas=(4, 8, 16)) -> dict:
    """Bin-width sensitivity, coverage growth, and a cross-estimator check."""
    y = corpus.properties[prop]
    sweep = []
    for w in widths:
        k = ladder_keys(corpus.netlists, bin_width=w)[rung]
        est = _floor_at(k, y, n_boot=200)
        sweep.append({"bin_width": w, "phi": est["phi"], "coverage": est["coverage"],
                      "n_classes": est["n_classes"]})
    coarser_not_lower = all(sweep[i]["phi"] <= sweep[i + 1]["phi"] + 1e-9
                            for i in range(len(sweep) - 1))
    keys = ladder_keys(corpus.netlists, bin_width=BIN_WIDTH)[rung]
    growth = []
    for frac in (0.25, 0.5, 1.0):
        m = int(len(corpus) * frac)
        growth.append({"n": m, "coverage": partition_floor(keys[:m], y[:m])["coverage"]})
    X = np.stack([_featurize(nl, rung) for nl in corpus.netlists])
    cross = [{"kappa": kk, "phi": float(neighborhood_floor(X, y, kappa=kk))}
             for kk in kappas]
    return {"property": prop, "rung": rung, "bin_width_sweep": sweep,
            "wider_bins_never_lower_floor": bool(coarser_not_lower),
            "coverage_growth": growth, "neighborhood_cross_check": cross,
            "partition_phi": _floor_at(keys, y, n_boot=200)["phi"]}


def e3_e6_decomposition(e2: dict, model_errors: Dict[str, Dict[str, float]]) -> dict:
    """Split each measured model error into floor and gap, with a propagated CI.

    A floor-limited verdict needs the gap CI to include zero; a gap-limited
    verdict needs it to exclude zero.
    """
    out = {}
    for prop, per_rung in model_errors.items():
        if prop not in e2:
            continue
        rows = {}
        for rung, mse in per_rung.items():
            if rung not in e2[prop]["curve"]:
                continue
            est = e2[prop]["curve"][rung]
            d = decomposition(est["phi"], float(mse))
            lo = float(mse) - est["ci_hi"]
            hi = float(mse) - est["ci_lo"]
            d.update(gap_ci=[lo, hi],
                     verdict=("floor-limited" if lo <= 0 <= hi else
                              "gap-limited" if lo > 0 else "inconsistent: gap CI below zero"))
            rows[rung] = d
        out[prop] = rows
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def emit_latex(e2: dict, stamp: str, path: Path) -> None:
    lines = [f"% SimBoundary results ({stamp}). Generated by simboundary_run.py.",
             "% Paste these rows into the main results table, replacing the placeholders.",
             "% Columns: property & rung & floor & coverage & verdict"]
    for prop, res in sorted(e2.items()):
        for rung in RUNGS:
            c = res["curve"][rung]
            if not np.isfinite(c["phi"]):
                continue
            lines.append(
                f"{prop.replace('_', ' ')} & $\\R_{rung[1]}$ & {c['phi']:.4g} "
                f"& {c['coverage']:.2f} & "
                f"{'below tol' if c['ci_hi'] < res['tolerance'] else 'above tol'} \\\\")
        lines.append("\\addlinespace[2pt]")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the SimBoundary experiments.")
    ap.add_argument("--corpus", help="JSON Lines corpus (see simboundary_data.py)")
    ap.add_argument("--corpus-b", help="second corpus, enables E5")
    ap.add_argument("--model-errors", help="JSON {property: {rung: mse}}, enables E3/E6")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--control-property", default=None,
                    help="property used as the E1 positive control")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit writing results from a synthetic corpus (stamped)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the whole pipeline on a synthetic corpus")
    a = ap.parse_args(argv)

    if a.self_test:
        return 0 if run_self_test() else 1
    if not a.corpus:
        ap.error("--corpus is required (or use --self-test)")

    try:
        corpus = load_jsonl(a.corpus)
    except (FileNotFoundError, SchemaError) as e:
        print(f"cannot load corpus:\n{e}")
        return 2
    if corpus.synthetic and not a.allow_synthetic:
        raise SystemExit("refusing to write results from a synthetic corpus; "
                         "pass --allow-synthetic if this is a pipeline test")
    stamp = corpus.stamp()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = ladder_keys(corpus.netlists, bin_width=BIN_WIDTH)

    control = a.control_property or corpus.property_names()[0]
    e1 = e1_positive_control(corpus, keys, control)
    print(f"E1 positive control on {control!r}: {'PASS' if e1['passed'] else 'FAIL'} "
          f"(floor={e1['floor']['phi']:.4g}, tol={e1['tolerance']:.4g}, "
          f"null ratio={e1['null_ratio']:.2f})")
    results = {"stamp": stamp, "corpus": corpus.name, "n": len(corpus),
               "tolerance_frac": TOLERANCE_FRAC, "E1": e1}
    if not e1["passed"]:
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"  why: {e1['diagnosis']}")
        print("E1 is the gate: stopping before E2 onward. See results.json.")
        return 2

    results["E2"] = e2 = e2_floor_curves(corpus, keys)
    results["E4"] = e4_collisions(corpus, keys)
    results["E7"] = e7_oracle_injection(corpus, keys, control)
    results["E8"] = e8_robustness(corpus, control)
    if a.corpus_b:
        cb = load_jsonl(a.corpus_b)
        kb = ladder_keys(cb.netlists, bin_width=BIN_WIDTH)
        results["E5"] = e5_cross_benchmark(e2, e2_floor_curves(cb, kb), corpus.name, cb.name)
    if a.model_errors:
        results["E3_E6"] = e3_e6_decomposition(e2, json.loads(Path(a.model_errors).read_text()))

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    emit_latex(e2, stamp, out / "simboundary_results.tex")
    for prop, r in sorted(e2.items()):
        print(f"  {prop:16s} min sufficient rung = {r['min_sufficient_rung'] or 'none'} "
              f"({r['verdict']})")
    print(f"\nwrote {out/'results.json'} and {out/'simboundary_results.tex'}  [{stamp}]")
    return 0


# ==========================================================================
# SELF-TEST: the whole pipeline on a synthetic corpus with a known answer
# ==========================================================================
def run_self_test() -> bool:
    import tempfile
    ok = True
    sd = 0.6
    corpus = make_synthetic_corpus(300, seed=BASE_SEED, hidden_noise_sd=sd)
    keys = ladder_keys(corpus.netlists, bin_width=BIN_WIDTH)

    e1 = e1_positive_control(corpus, keys, "p_graph")
    print(f"  [{'ok' if e1['passed'] else 'FAIL'}] E1 gate passes for the "
          f"graph-determined label (floor={e1['floor']['phi']:.4g}, "
          f"tol={e1['tolerance']:.4g})")
    ok &= e1["passed"]

    e2 = e2_floor_curves(corpus, keys)
    mono = all(r["monotone_nonincreasing"] for r in e2.values())
    print(f"  [{'ok' if mono else 'FAIL'}] E2 floor curves are non-increasing "
          f"along the ladder (Proposition 1)")
    ok &= mono

    got, want = e2["p_hidden"]["curve"]["R4"]["phi"], sd ** 2
    close = abs(got - want) < 0.35 * want
    print(f"  [{'ok' if close else 'FAIL'}] E2 recovers the injected hidden variance "
          f"at R4: {got:.4g} vs known {want:.4g}")
    ok &= close

    e7 = e7_oracle_injection(corpus, keys, "p_graph", sds=(0.0, 0.4, 0.8))
    print(f"  [{'ok' if e7['slope_near_one'] else 'FAIL'}] E7 floor tracks injected "
          f"variance with slope {e7['slope']:.2f} (want ~1)")
    ok &= e7["slope_near_one"]

    e8 = e8_robustness(corpus, "p_graph")
    print(f"  [{'ok' if e8['wider_bins_never_lower_floor'] else 'FAIL'}] E8 wider "
          f"quantization never lowers the floor")
    ok &= e8["wider_bins_never_lower_floor"]

    dec = e3_e6_decomposition(e2, {"p_hidden": {"R4": e2["p_hidden"]["curve"]["R4"]["phi"] + 0.5}})
    v = dec["p_hidden"]["R4"]
    print(f"  [{'ok' if v['verdict'] == 'gap-limited' else 'FAIL'}] E3/E6 labels a model "
          f"error well above the floor as {v['verdict']}")
    ok &= v["verdict"] == "gap-limited"

    with tempfile.TemporaryDirectory() as d:
        emit_latex(e2, "SYNTHETIC", Path(d) / "r.tex")
        txt = (Path(d) / "r.tex").read_text()
        good = "SYNTHETIC" in txt and "\\R_4" in txt
        print(f"  [{'ok' if good else 'FAIL'}] LaTeX rows emitted and stamped SYNTHETIC")
        ok &= good

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
