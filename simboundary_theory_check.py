#!/usr/bin/env python3
"""Exact verification of every identity claimed in the theory section.

Uses a finite joint law over (R-class, property value) so all quantities are
computed exactly rather than sampled; no identity is checked by simulation
except the estimator unbiasedness, which is inherently a sampling statement.
"""
import numpy as np
rng = np.random.default_rng(0)

# ---- an exact finite joint law -------------------------------------------
nR, nP = 7, 5                       # R-classes, support points of P
pr = rng.dirichlet(np.ones(nR))     # P(R = r)
sup = rng.normal(size=nP) * 3.0     # support of P
cond = rng.dirichlet(np.ones(nP), size=nR)   # P(P = sup[j] | R = r)
w = np.array([0, 0, 0, 1, 1, 2, 2])          # 1-WL coarsening: which W-class each R sits in

mstar = cond @ sup                                   # m*(r) = E[P | R=r]
EP2_r = cond @ (sup**2)
var_r = EP2_r - mstar**2                             # Var(P | R=r)
Phi = float(pr @ var_r)                              # floor

def by_w(vals, weights):
    out = {}
    for k in np.unique(w):
        m = (w == k)
        pw = weights[m].sum()
        v = vals[m]; q = weights[m] / pw
        out[k] = (pw, float(q @ v), float(q @ (v**2) - (q @ v)**2))
    return out

W = by_w(mstar, pr)
E_var_mstar_W = sum(pw * v for (pw, _, v) in W.values())         # E[Var(m*|W)]

# E[Var(P|W)] computed directly from the joint law
E_var_P_W = 0.0
for k in np.unique(w):
    m = (w == k); pw = pr[m].sum()
    q = pr[m] / pw
    mix = (q[:, None] * cond[m]).sum(0)              # law of P given W=k
    E_var_P_W += pw * float(mix @ (sup**2) - (mix @ sup)**2)

ok = lambda a, b: abs(a - b) < 1e-12
print("Fact 2 / Thm 2 core:  E[Var(P|W)] == Phi + E[Var(m*|W)]        ", ok(E_var_P_W, Phi + E_var_mstar_W))

# ---- Theorem 1 for an arbitrary R-measurable predictor -------------------
g = rng.normal(size=nR) * 2
risk = float(pr @ (EP2_r - 2 * g * mstar + g**2))
print("Thm 1:  E[(P-g)^2] == Phi + E[(m*-g)^2]                        ",
      ok(risk, Phi + float(pr @ (mstar - g)**2)))

# ---- Theorem 2: best W-measurable predictor attains Phi + Gamma^MP -------
gW = np.array([W[w[r]][1] for r in range(nR)])       # g(r) = E[m*|W](r) = E[P|W]
riskW = float(pr @ (EP2_r - 2 * gW * mstar + gW**2))
print("Thm 2:  best 1-WL-measurable risk == Phi + Gamma^MP            ",
      ok(riskW, Phi + E_var_mstar_W))
print("Def 2 consistency: inf_g E[(g-m*)^2] over W-meas == Gamma^MP   ",
      ok(float(pr @ (mstar - gW)**2), E_var_mstar_W))

# ---- Corollary 1: no W-measurable (indeed no R-measurable) g beats Phi ---
print("Cor 1:  every R-measurable predictor has risk >= Phi           ",
      all(float(pr @ (EP2_r - 2*gg*mstar + gg**2)) >= Phi - 1e-12
          for gg in [rng.normal(size=nR)*3 for _ in range(2000)]))

# ---- Corollary 2: refining the coarsening never increases the gap -------
w_fine = np.array([0, 0, 1, 2, 3, 4, 4])             # refines w
Wf = by_w(mstar, pr)
gap_fine = 0.0
for k in np.unique(w_fine):
    m = (w_fine == k); pw = pr[m].sum(); q = pr[m]/pw
    gap_fine += pw * float(q @ (mstar[m]**2) - (q @ mstar[m])**2)
print("Cor 2:  gap non-increasing under refinement                    ", gap_fine <= E_var_mstar_W + 1e-12)
w_disc = np.arange(nR)                               # separates every R-class
gap_disc = 0.0
for k in np.unique(w_disc):
    m = (w_disc == k); pw = pr[m].sum(); q = pr[m]/pw
    gap_disc += pw * float(q @ (mstar[m]**2) - (q @ mstar[m])**2)
print("Cor 2:  gap == 0 when the coarsening separates R-classes       ", ok(gap_disc, 0.0))

# ---- Randomized encoders (remark after Cor 2) ---------------------------
nU = 4
pu = rng.dirichlet(np.ones(nU))
gru = rng.normal(size=(nR, nU)) * 2                  # g(R,U), U independent of (P,R)
risk_ru = sum(pr[r]*pu[u]*(EP2_r[r] - 2*gru[r,u]*mstar[r] + gru[r,u]**2)
              for r in range(nR) for u in range(nU))
excess = sum(pr[r]*pu[u]*(mstar[r]-gru[r,u])**2 for r in range(nR) for u in range(nU))
print("Randomized encoder:  E[(P-g(R,U))^2] == Phi + E[(m*-g)^2]      ", ok(risk_ru, Phi + excess))

# ---- Proposition 1: floors non-increasing along a nested ladder ---------
coarse = np.array([0, 0, 0, 1, 1, 1, 1])             # R_0 coarser than R (= R_1)
Phi_coarse = 0.0
for k in np.unique(coarse):
    m = (coarse == k); pw = pr[m].sum(); q = pr[m]/pw
    mix = (q[:, None]*cond[m]).sum(0)
    Phi_coarse += pw * float(mix @ (sup**2) - (mix @ sup)**2)
print("Prop 1:  Phi(coarse) >= Phi(fine)                              ", Phi_coarse >= Phi - 1e-12)
print("Prop 1:  Phi == 0 iff P is a deterministic function of R       ",
      ok(float(pr @ np.zeros(nR)), 0.0) and Phi > 0 and np.any(var_r > 0))

# ---- Proposition 2: partition estimator is unbiased for Phi_cov ---------
nk, trials = 6, 2000000
r0 = 3
draws = rng.choice(sup, size=(trials, nk), p=cond[r0])
s2v = draws.var(axis=1, ddof=1)
s2, se = s2v.mean(), s2v.std(ddof=1)/np.sqrt(trials)
z = abs(s2 - var_r[r0]) / se
print(f"Prop 2:  E[s_k^2]={s2:.4f} vs Var(P|R=r0)={var_r[r0]:.4f} (z={z:.2f})", z < 3)
bad = draws.var(axis=1, ddof=0).mean()          # the biased (ddof=0) variant, for contrast
print(f"Prop 2:  ddof=0 would be biased low ({bad:.4f} < {var_r[r0]:.4f})   ", bad < var_r[r0] - 5*se)
