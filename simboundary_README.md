# SimBoundary — Student README

**Paper:** *Beyond Expressiveness: A Representation-Sufficiency Floor for GNN-Based Analog Circuit Performance Prediction*

This is your entry point. It explains, in plain terms, what the paper claims, how to build it, what experiments to run, and what to do next. The detailed experimental protocol lives in `simboundary_runbook.txt`; this file points you there.

---

## 1. What the paper claims and why it matters

**The one-sentence thesis.** A graph neural network (GNN) can only ever be as good as the graph you feed it, so for each circuit metric there is a *floor* on how well any GNN can predict it from a given graph representation. Measuring that floor along a ladder of representations tells us, per metric, the cheapest graph that is sufficient, and whether any metric needs so much detail that a GNN offers no shortcut over simulation.

**The gap in the field.** Everyone agrees a circuit is a graph and GNNs are the natural model. The current state of the art, FALCON (NeurIPS 2025), predicts a broad panel of metrics from a million-circuit dataset; the RF state of the art (Asadi et al. 2025) compares net-level, component-level, and terminal-level graphs. The field improves prediction on two fronts: better encoders (flow attention, higher-order GNNs) and richer graph abstractions. Both aim at the model. But the field's own strongest results already report, without explaining, that accuracy is governed by how well the representation matches each metric's physics, and that some metrics stay hard no matter the architecture. Those are two different limits with opposite fixes:

- **Encoder expressiveness** is how well a GNN tells apart the graphs it is given. Better architectures improve this. This is what most work targets.
- **Representation sufficiency** is whether the graph *contains* the information at all. If two circuits are the *same* annotated graph but differ in a metric, no architecture can tell them apart, because they are the same input. Better encoders are then **provably useless**; only a richer representation helps, and taken far enough that means adding back what the simulator computes.

**The contribution (delta over the closest work).** We prove these two limits *add* (a clean Pythagorean split of any GNN's error into a **sufficiency floor** plus a **closable gap**), we turn the representations the field actually debates into a **measuring ladder**, and we measure, per metric, the **minimum representation** at which the floor vanishes. This explains the RF SOTA's "align the representation with the physics" observation by turning it into a measured quantity. The closest neighbors each touch the phenomenon without isolating it: Flow-Attentional GNNs (TMLR 2025) and the RF-informed GNN (2025) work on the *gap* (one by a better encoder, one by a richer representation) and report per-metric error that conflates floor and gap; a July-2025 graph-sufficiency theory paper proves expressive layers preserve the *input's* information, which actually strengthens our point (the operative limit is then the *representation's* sufficiency, i.e. our floor). See Table 1 in the paper for the row-by-row delta.

**One honesty point built into the framing.** On a clean dataset whose labels are a deterministic function of the sized netlist, every floor may fall to zero once the representation is detailed enough; then the result is the per-metric *minimum sufficient rung*, which is still novel and useful. A floor that *persists* at the richest representation requires the labels to vary because of something no graph rung captures (layout parasitics, process corners, device mismatch). Which metrics fall where is measured by the experiments, not assumed. Do not write the paper as if "metric X fundamentally needs simulation" is known before the runs.

**What this bundle is and is NOT.** It is a complete, compiling paper with the theory proved in full, an instrument, and a lower bound that is correct by construction. It is **not** a finished empirical result: every number in the PDF is a visible purple `[RESULT]` placeholder. The paper becomes submittable only after you run the experiments in the runbook and fill those in.

---

## 1b. Every file in this bundle

Paper
- `main.tex`, `simboundary.bib`, `main.bbl`, `main.pdf` — the paper; build with
  `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- `simboundary_submission.zip` — the four paper files, nothing else, for upload.

Code you need to run the experiments (NumPy only, no ML dependencies)
- `simboundary_ladder.py` — builds the R0..R4 graphs from a netlist and returns the
  canonical key per rung. Also checks the nesting that Proposition 1 assumes.
- `simboundary_data.py` — the corpus interchange format, strict loaders, and the
  clearly-labelled synthetic generator. No silent fallback to fake data.
- `simboundary_convert.py` — takes a benchmark release to that format. The whole
  pipeline is written and tested; you supply one parser function for your
  benchmark. Reports per-rung class coverage so you can see before running the
  protocol whether the corpus has enough repeated classes to measure a floor.
- `simboundary_analysis.py` — the estimators: partition floor, bootstrap CI,
  permutation null, neighborhood floor, floor/gap decomposition.
- `simboundary_run.py` — the driver. Runs E1, E2, E4, E5, E7, E8 and computes the
  E3/E6 decomposition from measured model errors. Writes `results.json` and
  `simboundary_results.tex`.
- `simboundary_theory_check.py` — verifies every identity in the theory section.

Paper quality control (optional, used while writing)
- `simboundary_figcheck.py` — word overlaps and strokes crossing text in the figures.
- `simboundary_lineclearance.py` — distance from each figure label to the nearest line.

Docs
- `simboundary_runbook.txt` — the experimental protocol in detail.
- `simboundary_README.md` — this file.
- `requirements.txt` — numpy is the only dependency for the experiments.
- `example_model_errors.json` — the format for `--model-errors` (E3/E6).

### Run order

    python3 simboundary_theory_check.py     # algebra of the theory
    python3 simboundary_analysis.py         # estimator smoke test
    python3 simboundary_ladder.py           # ladder self-test
    python3 simboundary_data.py             # loader self-test
    python3 simboundary_run.py --self-test  # whole pipeline on synthetic data
    python3 simboundary_convert.py --demo   # conversion, end to end

All six must pass before touching real data (about 10 seconds in total). Then convert a benchmark. Write your parser in `simboundary_convert.py`
(`parse_example` is a complete worked one), check it on a handful of records,
convert, then run the protocol:

    python3 simboundary_convert.py --in raw.jsonl --parser falcon --dry-run 5
    python3 simboundary_convert.py --in raw.jsonl --parser falcon --out falcon.jsonl

The converter prints per-rung coverage. If coverage is near zero at a rung, the
partition estimator cannot measure the floor there; report that rather than
widening the quantization bin until collisions appear. Then:

    python3 simboundary_run.py --corpus falcon.jsonl --control-property dc_power_w \
        --out results/
    # add --corpus-b ocb.jsonl for E5, and --model-errors errors.json for E3/E6

`results/simboundary_results.tex` holds the rows that replace the placeholders in
the paper's results table.

## 2. How to read and build the paper

**Files**
| File | What it is |
|---|---|
| `main.tex` | the paper source (venue-neutral single-column) |
| `simboundary.bib` | bibliography (verified entries + `% VERIFY` flags) |
| `main.bbl` | precompiled bibliography (fallback so one `pdflatex` pass resolves refs) |
| `main.pdf` | the built paper |
| `simboundary_submission.zip` | self-contained source for upload |
| `simboundary_analysis.py` | floor estimators + smoke test (numpy only) |
| `simboundary_runbook.txt` | the detailed experiment protocol |
| `simboundary_README.md` | this file |

**Build command** (four passes, as usual):
```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```
It compiles with zero errors, zero undefined citations, and zero overfull boxes on a standard TeX Live.

**Where each claim lives**
- The two-limit idea and the teaser: Figure 1 + Section 1.
- The formal split: Theorem 1 (Section 4), proof in Appendix A.
- Encoder-independence of the floor: Corollary 1; the role of 1-WL: Remark 1.
- The representation ladder: Table 2.
- The estimators (what the code implements): Section 5 + Algorithm 1.
- The experiments and the placeholder results: Section 6, Table 4, Figure 3.

**Venue.** Written to drop into either target:
- **ICLR 2027** (primary; A*): deadline ~late September 2026. Best fit for a representation-limits + "when learning replaces simulation" paper. Swap in the ICLR style file and enable line numbers (the preamble already contains the line-number-overlap fix).
- **LoG 2026** (Learning on Graphs; strong alternative, earlier deadline ~late August 2026): the graph-ML-native venue, famously constructive reviews (ideal for a first submission), 9-page PMLR-archival. Swap in the LoG style.
- Not AAAI-27: its deadline (Jul 20/27, 2026) is too soon to run the experiments honestly.
Pick one archival venue — you cannot submit the same archival paper to both at once.

---

## 3. What the experiments are and how to run them

Full detail is in `simboundary_runbook.txt`. The essentials:

**Do this first (a few minutes, CPU).** Pilot `P0`: compute the gain floor at rungs R0–R2 on Ckt-Bench-101 with no training, just to confirm your keys and labels are sane, before you download FALCON or touch a GPU.

**Then the positive control, `E1`.** Predict the most reliably easy metric on each benchmark: **DC power on FALCON** (reported well under 1% error, so essentially a function of the sized graph) and **DC gain on OCB** (op-amp gain is well predicted). The framework says these should be *easy* (low floor, a plain GIN nearly matches it). **If E1 does not come out easy, your pipeline is broken; stop and debug before trusting anything else.** Note: do NOT use FALCON gain-type metrics as the control, because the RF SOTA reports them as among the *harder* metrics; that is part of what E2 studies, not a control.

**Then the main results, `E2`/`E3`.** Floor curves across R0–R4 for a spread of metrics (FALCON: gain-type, bandwidth, noise figure, DC power; OCB: gain, bandwidth, phase margin, FoM), plus the best GNN error at each rung. The primary output is the per-metric **minimum sufficient rung** (the coarsest representation at which the floor reaches tolerance). `E4` extracts concrete "same graph, different SPICE" circuit pairs and explains the physics. `E5` checks that the per-metric pattern agrees across FALCON and OCB.

**The baselines are the current SOTA, not toy GNNs.** The encoder ladder (used only to measure the gap) spans three tiers: 1-WL message passing (GIN, D-VAE, DAGNN); the strongest current circuit predictors (FALCON's net-level GNN, the RF-informed terminal-level GENConv, Flow-Attention, and the pretrained encoder DICE); and beyond-message-passing models (a subgraph GNN and the GraphGPS graph transformer, a universal function approximator on graphs). The graph transformer matters: Theorem 2 bounds only message-passing GNNs, so a reviewer will ask whether a transformer beats the floor. It cannot (Corollary 1), and E3/E7 show even GraphGPS plateaus at the floor for floor-limited metrics.

**Three experiments carry the venue-level weight.** `E6` (the justifying experiment) reproduces the SOTA leaderboard numbers and decomposes them, showing at least one metric whose competitive total error is actually floor-limited (so "improve the architecture" is misdirected) and another with similar total error that is gap-limited. `E7` (oracle validation) injects a known amount of non-netlist variation and verifies the floor tracks it and no encoder beats it, validating the simulation-limit proposition end to end. `E8` (estimator robustness) checks the floor is stable across bin widths, agrees with a neighborhood and an information-theoretic cross-estimator, converges with coverage, and is not a label-noise artifact.

**Compute.** The floor analysis (E2, the headline) is training-free and runs on CPU over the full million-circuit set. The expensive part is training the encoder ladder; reproduce FALCON/RF-informed from their released code and use DICE's released weights rather than training from scratch, and stage the ladder if GPU-limited.

**What "promising" looks like.** E1 passes; the floor curves separate by metric, so different metrics have visibly different minimum sufficient rungs (some resolved at a coarse rung, some only at a fine one), with confidence intervals that make the ordering real. If in addition some metric's floor persists at the richest rung, that is the stronger "no GNN shortcut over simulation" result, but it is a bonus, not a requirement.

**What "negative" looks like, and that's OK.** If every metric's floor reaches tolerance by R4 on both benchmarks, the honest finding is the per-metric minimum sufficient rung ("for these families the graph at full detail is sufficient; the open problem is the gap and the cheapest sufficient input"). That is still a real, publishable measurement. If floors are dominated by low coverage everywhere and the two estimators disagree, you cannot claim a clean result; report the limitation and consider a denser-collision subset. Do not force a "requires simulation" story the data does not support.

---

## 4. What the code computes and how to run it

`simboundary_analysis.py` is numpy-only and implements exactly the paper's quantities:
- `graph_key(adj, node_feats)` — the representation-equivalence key via 1-WL color refinement (permutation-invariant).
- `partition_floor(keys, y)` — the sufficiency floor `E[Var(P|R)]` from **data only** (Eq. 2), with coverage.
- `bootstrap_floor_ci(...)`, `permutation_null(...)` — confidence interval and a null that should return the total variance.
- `neighborhood_floor(X, y, kappa)` — an approximate-collision cross-check for fine rungs (chunked, so it scales to 10k–50k circuits).
- `decomposition(floor, best_gnn_error)` — the gap accounting.

Run the self-validating smoke test (synthetic data with a **known** floor):
```
python3 simboundary_analysis.py
```
It should print `SMOKE TEST PASSED`. Every check compares an estimate to a known target — it validates the *method*, and deliberately contains **no** circuit results and asserts **no** paper outcome.

**The single most important usage rule:** the floor is computed from SPICE labels grouped by representation key, **never** from a model's residuals. If you compute a "floor" from a trained network, you measured the gap and broke the paper's core claim. Trained models are used only to see how close achievable error gets to the data-derived floor.

---

## 4b. Theory verification (`simboundary_theory_check.py`)

Run `python3 simboundary_theory_check.py`. It builds an explicit finite joint law over
(representation class, property value) and checks every identity the theory section
claims, exactly rather than by simulation:

- Theorem 1, the decomposition `E[(P-g)^2] = Phi + E[(m*-g)^2]` for an arbitrary predictor;
- Fact 2 / the core of Theorem 2, `E[Var(P|W)] = Phi + E[Var(m*|W)]`;
- Theorem 2, that the best 1-WL-measurable predictor attains `Phi + Gamma^MP`, and that this
  agrees with the Definition-2 form `inf_g E[(g-m*)^2]`;
- Corollary 1, that no predictor factoring through R beats the floor;
- Corollary 2, that the gap is non-increasing under refinement and vanishes once the
  coarsening separates the R-classes, while the floor is untouched;
- the randomized-encoder remark, `E[(P-g(R,U))^2] = Phi + E[(m*-g)^2]` for U independent of (P,R);
- Proposition 1, floor monotonicity along a nested ladder;
- Proposition 2, that the Bessel-corrected within-class variance is unbiased for the class
  conditional variance (and that the `ddof=0` variant is biased low, which is why the code
  uses `ddof=1`).

This is a check on the algebra, not a substitute for proof. Two things it cannot do: it
cannot validate the modelling assumptions (that message passing is 1-WL bounded, that the
ladder is genuinely nested, that the finest rung encodes the netlist), and it cannot replace
independent human verification of the written proofs, which remains a standing blocker.

## 5. Integrity and next-steps checklist

Before you submit, tick every box:

- [ ] **Confirm novelty** — the circuit-GNN and GNN-expressiveness areas move monthly (FALCON is NeurIPS 2025, RF-informed and graph-sufficiency are mid-2025). Re-run the deep arXiv search right before submission; if a floor/gap-for-circuits paper appeared, re-angle honestly.
- [ ] **Citations are verified** — the bibliography now has no `% VERIFY` flags; the previously-uncertain ones are confirmed (Graph of Circuits is NeurIPS **2023** by Shahane et al., not 2024; LaMAGIC is ICML 2024 by Chang et al.; the irreducible-error reference is Yuan and Lozano-Durán 2025). Re-check every citation against its source at submission time anyway; never invent authors.
- [ ] **Confirm the FALCON release** — check the exact field names, units, and redistribution license in the FALCON repo before using it. Do not assume the noise-figure or DC-power fields exist until you have seen them in the files.
- [ ] **Independently verify the proof** — Appendix A is short and classical but AI-drafted. A human author must check it.
- [ ] **Run E1 and believe it** — no headline claim until the positive control passes.
- [ ] **Replace every `[RESULT]`** — with numbers produced only by your runs.
- [ ] **Reconcile every claim** — abstract, title framing, and contributions must match the actual floor curves; in particular, do not keep any "requires simulation" phrasing for a metric whose floor turns out to vanish at a fine rung.
- [ ] **Get 2–3 independent human mock reviews** — and treat every issue they raise as blocking. This is the step most correlated with turning a strong draft into an accept, and it is the one this bundle cannot do for you.

**Honest bottom line.** The wedge is real and open, the theory is correct by construction, and the experiments are cheap and mostly training-free — a good position. But acceptance is decided by whether the floor/gap split actually shows up in the data and survives expert review, and that is your runs and your reviewers, not this draft.

## Integrity audit (added)

A full audit of the manuscript removed one verbatim quotation and several
per-metric numbers attributed to prior work that could not be verified against
the sources, corrected four theory statements (a missing countability
assumption in Theorem 2, an implicit injectivity assumption in Proposition 1,
the treatment of randomized encoders in Corollary 2, and the direction of the
confidence guard), and noted that the neighbourhood estimator is biased upward.
See the block at the end of `simboundary_runbook.txt` for the five items that
must be resolved before submission.
