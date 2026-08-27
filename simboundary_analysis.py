#!/usr/bin/env python3
# ==========================================================================
# simboundary_analysis.py
#
# Reference implementation of the quantities in
#   "When Can a Graph Replace a Simulator? A Representation-Sufficiency
#    Boundary for GNN-Based Analog Circuit Performance Prediction"
#
# Implements (Algorithm 1 in the paper):
#   * representation-equivalence keys via 1-WL color refinement on a
#     featured graph  (wl_refine, graph_key)
#   * partition estimator of the sufficiency floor  E[Var(P | R)]
#     with coverage, bootstrap CI, and a permutation null   (partition_floor,
#     bootstrap_floor_ci, permutation_null)
#   * neighborhood (approximate-collision) floor estimator  (neighborhood_floor)
#   * the two-level accounting  gap = best_GNN_error - floor  (decomposition)
#
# IMPORTANT (anti-fabrication):
#   This file COMPUTES estimators and VALIDATES them on SYNTHETIC data with a
#   KNOWN ground-truth floor. It contains NO circuit results and asserts NO
#   empirical outcome from the paper. Real floors come only from running these
#   functions on the Open Circuit Benchmark, per the runbook.
#
# Dependencies: numpy only.
# Usage:  python3 simboundary_analysis.py           # runs the smoke test
# ==========================================================================

from __future__ import annotations
import numpy as np
from collections import defaultdict
import hashlib

BASE_SEED = 0


# --------------------------------------------------------------------------
# 1. Representation-equivalence keys: 1-WL color refinement on featured graphs
# --------------------------------------------------------------------------
def wl_refine(adj: np.ndarray, node_feats, iters: int = 3):
    """Weisfeiler-Leman color refinement on a featured graph.

    adj        : (n,n) 0/1 adjacency (undirected or directed; we symmetrize
                 the neighbor multiset by including in- and out-neighbors).
    node_feats : length-n list of hashable initial node labels (e.g. component
                 type + quantized value + terminal role, depending on the rung).
    Returns a canonical multiset hash of the final color histogram, which is
    invariant to node permutation. Two graphs sharing this hash are declared
    R-equivalent (up to WL's resolution, i.e. the standard MPNN resolution).
    """
    n = len(node_feats)
    colors = [_h(str(f)) for f in node_feats]
    for _ in range(iters):
        new = []
        for v in range(n):
            neigh = sorted(colors[u] for u in range(n) if adj[v, u] or adj[u, v])
            new.append(_h(colors[v] + "|" + ",".join(neigh)))
        # Colors must stay content-bearing. An earlier version relabelled them
        # to small ints each round, which made the final histogram depend only
        # on the SHAPE of the colour partition: ['nmos','pmos'] and
        # ['res','cap'] then collided. That merges circuits that are not
        # R-equivalent, inflating within-class variance and so biasing the
        # floor upward, the direction that would falsely declare a property
        # simulation-required. Do not reintroduce a relabelling here.
        colors = new
    hist = tuple(sorted(_count(colors).items()))
    return _h(str(hist))


def graph_key(adj, node_feats, iters: int = 3) -> str:
    """Public alias: canonical R-equivalence key for one circuit graph."""
    return wl_refine(np.asarray(adj), list(node_feats), iters=iters)


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


# --------------------------------------------------------------------------
# 2. Partition estimator of the sufficiency floor  Phi = E[Var(P | R)]
# --------------------------------------------------------------------------
def partition_floor(keys, y):
    """Pooled within-equivalence-class variance estimator, Eq. (2).

    keys : length-N iterable of R-equivalence keys (e.g. from graph_key).
    y    : length-N array of property values (from SPICE).

    Returns dict with:
      phi        : pooled within-class variance  (the floor estimate)
      coverage   : fraction of samples in non-singleton classes
      n_classes  : number of non-singleton classes used
      total_var  : sample variance of y (for context / permutation checks)
    """
    y = np.asarray(y, dtype=float)
    groups = defaultdict(list)
    for k, val in zip(keys, y):
        groups[k].append(val)

    num = 0.0        # sum_k (n_k - 1) s_k^2
    den = 0.0        # sum_k (n_k - 1)
    covered = 0
    n_used = 0
    for vals in groups.values():
        nk = len(vals)
        if nk >= 2:
            sk2 = np.var(vals, ddof=1)
            num += (nk - 1) * sk2
            den += (nk - 1)
            covered += nk
            n_used += 1
    phi = num / den if den > 0 else float("nan")
    return {
        "phi": phi,
        "coverage": covered / len(y) if len(y) else 0.0,
        "n_classes": n_used,
        "total_var": float(np.var(y, ddof=1)) if len(y) > 1 else float("nan"),
    }


def bootstrap_floor_ci(keys, y, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = BASE_SEED):
    """Class-resampled bootstrap CI for the partition floor.

    Resamples whole equivalence classes with replacement (respecting the
    dependence structure), recomputing Phi each time.
    """
    keys = list(keys)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for k, val in zip(keys, y):
        groups[k].append(val)
    class_ids = list(groups.keys())
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(class_ids), size=len(class_ids), replace=True)
        bk, by = [], []
        for idx in pick:
            cid = class_ids[idx]
            vals = groups[cid]
            for v in vals:
                bk.append(cid)          # class label preserved within resample
                by.append(v)
        est = partition_floor(bk, by)["phi"]
        if np.isfinite(est):
            boots.append(est)
    if not boots:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)


def permutation_null(keys, y, n_perm: int = 500, seed: int = BASE_SEED):
    """Permutation null: shuffle y against keys. Under no representation-property
    link the floor should approach the total variance of y. Returns the mean
    permuted floor and its ratio to total variance (should be ~1.0)."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    tv = np.var(y, ddof=1)
    ests = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        e = partition_floor(keys, yp)["phi"]
        if np.isfinite(e):
            ests.append(e)
    mean_est = float(np.mean(ests)) if ests else float("nan")
    return {"mean_perm_floor": mean_est, "ratio_to_total_var": mean_est / tv if tv > 0 else float("nan")}


# --------------------------------------------------------------------------
# 3. Neighborhood estimator (approximate collisions at fine rungs)
# --------------------------------------------------------------------------
def neighborhood_floor(X: np.ndarray, y, kappa: int = 8):
    """Local-variance floor estimate under a smoothness assumption on m*.

    X : (N,d) representation embeddings (e.g. from a fixed untrained encoder,
        or a numeric feature vector). y : (N,) property values.
    For each point, average the variance of y over its kappa nearest neighbors
    (in Euclidean distance), then average across points. This estimates
    E[Var(P|R)] when m* is Lipschitz in the embedding. Report kappa-sensitivity
    in practice; use only as a cross-check on partition_floor.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(y)
    kappa = min(kappa, N - 1)
    sq = (X * X).sum(1)
    local_vars = np.empty(N)
    block = 512  # row-block size: memory is O(block * N), not O(N^2)
    for start in range(0, N, block):
        stop = min(start + block, N)
        # squared distances from rows [start:stop] to all points
        D2 = sq[start:stop, None] + sq[None, :] - 2.0 * (X[start:stop] @ X.T)
        for r, i in enumerate(range(start, stop)):
            D2[r, i] = np.inf  # exclude self
            nn = np.argpartition(D2[r], kappa)[:kappa]
            vals = np.append(y[nn], y[i])
            local_vars[i] = np.var(vals, ddof=1)
    return float(np.mean(local_vars))


# --------------------------------------------------------------------------
# 4. Two-level accounting: gap = best achievable GNN error - floor
# --------------------------------------------------------------------------
def decomposition(floor: float, best_gnn_error: float):
    """Return the expressiveness gap given a measured floor and the best
    achievable GNN test error (both as MSE on the same scale). Gap must be
    >= 0 up to estimation noise; a materially negative gap signals that the
    floor estimate is too high (low coverage) or the errors are on a different
    scale, and should be investigated, not reported."""
    gap = best_gnn_error - floor
    return {"floor": floor, "best_gnn_error": best_gnn_error, "gap": gap,
            "gap_nonneg_ok": gap >= -1e-6}


# ==========================================================================
# SMOKE TEST  (synthetic data with KNOWN ground-truth floor)
# ==========================================================================
def _make_synthetic(n=6000, n_classes=60, sigma_within=0.5, seed=BASE_SEED):
    """Property P = mu[class] + sigma_within * eps, eps ~ N(0,1).
    Then Var(P | class) = sigma_within^2 exactly, so the true floor at the
    'fine' representation (= class label) is sigma_within^2, regardless of the
    class means mu. This is the ground truth the partition estimator must
    recover. We also build a 'coarse' representation that merges class pairs;
    its true floor is sigma_within^2 + (within-merged-pair variance of mu),
    which must be >= the fine floor (ladder monotonicity)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 2.0, size=n_classes)          # class means (between-class signal)
    cls = rng.integers(0, n_classes, size=n)
    y = mu[cls] + sigma_within * rng.normal(size=n)
    fine_keys = [f"c{c}" for c in cls]               # fine representation
    coarse_keys = [f"m{c // 2}" for c in cls]        # merges classes in pairs
    # true coarse-floor signal component: E[Var(mu | merged)] on the sample.
    # (Total true coarse floor = sigma_within^2 + this signal component.)
    coarse_signal_var = _empirical_conditional_var(coarse_keys, mu[cls])
    return y, fine_keys, coarse_keys, sigma_within ** 2, coarse_signal_var, mu, cls


def _empirical_conditional_var(keys, signal):
    """E[Var(signal | keys)] on the sample (used to get the true coarse floor's
    signal component; the noise component sigma_within^2 adds on top)."""
    g = defaultdict(list)
    for k, s in zip(keys, signal):
        g[k].append(s)
    num = den = 0.0
    for vals in g.values():
        nk = len(vals)
        if nk >= 2:
            num += (nk - 1) * np.var(vals, ddof=1)
            den += (nk - 1)
    return num / den if den > 0 else 0.0


def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST  (synthetic; validates estimators against known floor)")
    print("=" * 70)
    ok = True
    sigma_within = 0.5
    y, fine, coarse, true_fine_floor, coarse_signal_var, mu, cls = _make_synthetic(
        sigma_within=sigma_within)
    true_coarse_floor = true_fine_floor + coarse_signal_var

    # --- (a) WL keys are permutation-invariant --------------------------------
    adj = np.array([[0, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0]])
    feats = ["R", "C", "M", "R"]
    perm = [2, 0, 3, 1]
    k1 = graph_key(adj, feats)
    k2 = graph_key(adj[np.ix_(perm, perm)], [feats[p] for p in perm])
    inv_ok = (k1 == k2)
    print(f"[a] WL key permutation-invariant: {inv_ok}")
    ok &= inv_ok

    # --- (b) partition estimator recovers the true fine floor -----------------
    res_fine = partition_floor(fine, y)
    err = abs(res_fine["phi"] - true_fine_floor)
    b_ok = err < 0.05
    print(f"[b] fine floor: est={res_fine['phi']:.4f}  true={true_fine_floor:.4f}  "
          f"|err|={err:.4f}  coverage={res_fine['coverage']:.2f}  -> {b_ok}")
    ok &= b_ok

    # --- (c) ladder monotonicity: coarse floor >= fine floor ------------------
    res_coarse = partition_floor(coarse, y)
    mono_ok = res_coarse["phi"] >= res_fine["phi"] - 1e-6
    print(f"[c] coarse floor est={res_coarse['phi']:.4f} (true~{true_coarse_floor:.4f}) "
          f">= fine floor est={res_fine['phi']:.4f} : {mono_ok}")
    ok &= mono_ok

    # --- (d) bootstrap CI brackets the true fine floor ------------------------
    lo, hi = bootstrap_floor_ci(fine, y, n_boot=400, seed=BASE_SEED)
    ci_ok = (lo <= true_fine_floor <= hi)
    print(f"[d] 95% bootstrap CI for fine floor = [{lo:.4f}, {hi:.4f}]  "
          f"brackets true {true_fine_floor:.4f}: {ci_ok}")
    ok &= ci_ok

    # --- (e) permutation null returns ~ total variance ------------------------
    perm_res = permutation_null(fine, y, n_perm=200, seed=BASE_SEED)
    null_ok = 0.9 <= perm_res["ratio_to_total_var"] <= 1.1
    print(f"[e] permutation-null floor / total-var = "
          f"{perm_res['ratio_to_total_var']:.3f} (expect ~1.0): {null_ok}")
    ok &= null_ok

    # --- (f) neighborhood estimator in the DENSE regime it is designed for ----
    # (In sparse / high-dim regimes it is biased UPWARD by local signal
    #  curvature; that is the smoothness caveat stated in the paper. Here we
    #  validate the implementation where neighbors are genuinely close.)
    rng = np.random.default_rng(BASE_SEED)
    N = 6000
    X = rng.uniform(0.0, 1.0, size=(N, 1))        # 1-D, dense
    noise_sd = 0.3
    yc = np.sin(3.0 * X[:, 0]) + noise_sd * rng.normal(size=N)  # smooth signal
    nb = neighborhood_floor(X, yc, kappa=8)
    nb_ok = abs(nb - noise_sd ** 2) < 0.02
    print(f"[f] neighborhood floor (dense) est={nb:.4f}  true noise var={noise_sd**2:.4f}: {nb_ok}")
    ok &= nb_ok

    # --- (g) decomposition accounting sanity ----------------------------------
    d = decomposition(floor=res_fine["phi"], best_gnn_error=res_fine["phi"] + 0.12)
    dec_ok = d["gap_nonneg_ok"] and abs(d["gap"] - 0.12) < 1e-6
    print(f"[g] decomposition: floor={d['floor']:.3f} best_err={d['best_gnn_error']:.3f} "
          f"gap={d['gap']:.3f}: {dec_ok}")
    ok &= dec_ok

    print("-" * 70)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    print("-" * 70)
    print("Note: these are synthetic validations of the METHOD. Real per-property")
    print("circuit floors are produced only by running these estimators on OCB")
    print("data as specified in simboundary_runbook.txt.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_smoke_test() else 1)
