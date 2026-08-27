#!/usr/bin/env python3
"""SimBoundary: the representation ladder R0..R4.

Turns a circuit netlist into the five graph abstractions of Table 1 and returns
a canonical key per rung. The keys are what the floor estimator partitions on,
so this module is where the paper's ladder becomes an actual measurement.

    R0  bare topology     connectivity only, no types, no values
    R1  typed topology    component-type labels
    R2  sized graph       + quantized device values, terminals still untyped
    R3  ported netlist    + terminal identity (gate/drain/source, +/-)
    R4  ported + flow     + branch orientation (Kirchhoff reference direction)

Graph model: a device-net incidence graph. Devices and nets are nodes. At R0-R2
a device connects directly to each net it touches. At R3-R4 each (device,
terminal) pair becomes its own port node carrying the terminal role, which is
how terminal-level circuit graphs are built in the literature; the role (and at
R4 the orientation) rides on the port node's label so that plain colour
refinement sees it.

NumPy only. Run `python3 simboundary_ladder.py` for the self-test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from simboundary_analysis import graph_key

RUNGS = ("R0", "R1", "R2", "R3", "R4")

# Reference current direction per terminal role, used only at R4. The sign is a
# convention: +1 means the branch reference current is defined into the device
# at that terminal. Any consistent convention works; what matters is that the
# same one is applied to every circuit.
DEFAULT_ORIENT: Dict[str, int] = {
    "d": +1, "c": +1, "p": +1, "in": +1, "+": +1,
    "s": -1, "e": -1, "n": -1, "out": -1, "-": -1,
    "g": 0, "b": 0,
}


@dataclass
class Device:
    """One circuit element.

    kind      : device type string, e.g. 'nmos', 'pmos', 'res', 'cap', 'ind'.
    terminals : terminal role -> net name, e.g. {'g': 'in', 'd': 'out', 's': 'gnd'}.
    value     : the device-defining scalar (W/L, resistance, capacitance...).
                None means the device carries no value (used at R2 as its own
                quantization bucket).
    orient    : optional per-role override of DEFAULT_ORIENT.
    """
    kind: str
    terminals: Dict[str, str]
    value: Optional[float] = None
    orient: Optional[Dict[str, int]] = None

    def role_orient(self, role: str) -> int:
        if self.orient and role in self.orient:
            return int(self.orient[role])
        return DEFAULT_ORIENT.get(role, 0)


@dataclass
class Netlist:
    """A circuit as a list of devices. Nets are implied by terminal net names."""
    devices: List[Device] = field(default_factory=list)
    name: str = ""

    def nets(self) -> List[str]:
        seen = []
        for d in self.devices:
            for net in d.terminals.values():
                if net not in seen:
                    seen.append(net)
        return seen


# --------------------------------------------------------------------------
# value quantization (R2 and above)
# --------------------------------------------------------------------------
def quantize(value: Optional[float], n_decades_per_bin: float = 0.25) -> str:
    """Log-scale bucket for a device value.

    Device values span orders of magnitude, so bins are fixed-width in log10.
    `n_decades_per_bin` is the pre-registered bin width; E8 sweeps it. A value
    of None or a non-positive value gets its own bucket rather than being
    silently dropped.
    """
    if value is None:
        return "none"
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        return f"nonpos:{v!r}"
    return f"q{int(np.floor(np.log10(v) / n_decades_per_bin))}"


# --------------------------------------------------------------------------
# graph construction, one builder per rung
# --------------------------------------------------------------------------
def build_graph(netlist: Netlist, rung: str, bin_width: float = 0.25):
    """Return (adjacency, node_labels) for one rung.

    Raises ValueError on an unknown rung rather than defaulting, so a typo
    cannot silently produce a coarser representation than intended.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")

    ported = rung in ("R3", "R4")
    nets = netlist.nets()
    net_index = {net: i for i, net in enumerate(nets)}

    labels: List[str] = []
    # net nodes first
    for _ in nets:
        labels.append("net" if rung != "R0" else "x")
    net_nodes = list(range(len(nets)))

    device_nodes: List[int] = []
    port_nodes: List[tuple] = []          # (node_id, device_idx, role)
    for di, dev in enumerate(netlist.devices):
        dnode = len(labels)
        device_nodes.append(dnode)
        if rung == "R0":
            labels.append("x")
        elif rung == "R1":
            labels.append(f"dev:{dev.kind}")
        else:
            labels.append(f"dev:{dev.kind}:{quantize(dev.value, bin_width)}")
        if ported:
            for role in sorted(dev.terminals):
                pnode = len(labels)
                if rung == "R3":
                    labels.append(f"port:{role}")
                else:  # R4 adds the branch reference direction
                    labels.append(f"port:{role}:{dev.role_orient(role):+d}")
                port_nodes.append((pnode, di, role))

    n = len(labels)
    adj = np.zeros((n, n), dtype=np.int8)

    def link(a: int, b: int) -> None:
        adj[a, b] = 1
        adj[b, a] = 1

    if not ported:
        for di, dev in enumerate(netlist.devices):
            for role, net in dev.terminals.items():
                link(device_nodes[di], net_nodes[net_index[net]])
    else:
        for pnode, di, role in port_nodes:
            link(device_nodes[di], pnode)
            link(pnode, net_nodes[net_index[netlist.devices[di].terminals[role]]])

    return adj, labels


def rung_key(netlist: Netlist, rung: str, bin_width: float = 0.25,
             wl_iters: int = 3) -> str:
    """Canonical key for one circuit at one rung. Equal keys = same class."""
    adj, labels = build_graph(netlist, rung, bin_width=bin_width)
    return graph_key(adj, labels, iters=wl_iters)


def ladder_keys(netlists: Sequence[Netlist], bin_width: float = 0.25,
                wl_iters: int = 3) -> Dict[str, List[str]]:
    """Keys for every circuit at every rung: {rung: [key per circuit]}."""
    return {r: [rung_key(nl, r, bin_width, wl_iters) for nl in netlists]
            for r in RUNGS}


def check_nesting(keys_by_rung: Dict[str, List[str]]) -> Dict[str, object]:
    """Test the nesting Proposition 1 assumes.

    Proposition 1 requires each rung to be a deterministic function of the next,
    so two circuits sharing a key at rung i+1 must share it at rung i. This is a
    property of the constructed keys, not something the proof can supply, so it
    is checked directly. Any violation is reported rather than absorbed.
    """
    n = len(next(iter(keys_by_rung.values())))
    violations = []
    for a, b in zip(RUNGS, RUNGS[1:]):
        ka, kb = keys_by_rung[a], keys_by_rung[b]
        for i in range(n):
            for j in range(i + 1, n):
                if kb[i] == kb[j] and ka[i] != ka[j]:
                    violations.append((b, a, i, j))
    return {"ok": not violations, "n_violations": len(violations),
            "examples": violations[:5]}


# ==========================================================================
# SELF-TEST
# ==========================================================================
def _inverter(w_n: float = 1.0, w_p: float = 2.0) -> Netlist:
    return Netlist([
        Device("nmos", {"g": "in", "d": "out", "s": "gnd"}, w_n),
        Device("pmos", {"g": "in", "d": "out", "s": "vdd"}, w_p),
    ], name="inverter")


def _diffpair(w: float = 1.0, r: float = 1e4) -> Netlist:
    return Netlist([
        Device("nmos", {"g": "inp", "d": "outn", "s": "tail"}, w),
        Device("nmos", {"g": "inn", "d": "outp", "s": "tail"}, w),
        Device("res", {"p": "vdd", "n": "outn"}, r),
        Device("res", {"p": "vdd", "n": "outp"}, r),
        Device("nmos", {"g": "vb", "d": "tail", "s": "gnd"}, 2 * w),
    ], name="diffpair")


def _random_netlist(rng: np.random.Generator) -> Netlist:
    nets = ["vdd", "gnd", "n1", "n2", "n3"]
    devs = []
    for _ in range(int(rng.integers(2, 6))):
        kind = str(rng.choice(["nmos", "pmos", "res", "cap"]))
        if kind in ("res", "cap"):
            t = {"p": str(rng.choice(nets)), "n": str(rng.choice(nets))}
        else:
            t = {"g": str(rng.choice(nets)), "d": str(rng.choice(nets)),
                 "s": str(rng.choice(nets))}
        devs.append(Device(kind, t, float(10 ** rng.uniform(-1, 4))))
    return Netlist(devs)


def run_self_test() -> bool:
    ok = True
    rng = np.random.default_rng(0)

    # 1. permutation invariance: reordering devices must not change any key
    nl = _diffpair()
    perm = Netlist([nl.devices[i] for i in [3, 0, 4, 1, 2]])
    for r in RUNGS:
        same = rung_key(nl, r) == rung_key(perm, r)
        print(f"  [{'ok' if same else 'FAIL'}] {r}: key invariant to device order")
        ok &= same

    # 2. each rung must actually separate what the rung below merges
    a, b = _inverter(1.0, 2.0), _inverter(1.0, 8.0)   # differ only in a value
    r1_same = rung_key(a, "R1") == rung_key(b, "R1")
    r2_diff = rung_key(a, "R2") != rung_key(b, "R2")
    print(f"  [{'ok' if r1_same else 'FAIL'}] R1 merges circuits differing only in device value")
    print(f"  [{'ok' if r2_diff else 'FAIL'}] R2 separates them once values are quantized")
    ok &= r1_same and r2_diff

    swap = Netlist([Device("nmos", {"g": "out", "d": "in", "s": "gnd"}, 1.0),
                    Device("pmos", {"g": "in", "d": "out", "s": "vdd"}, 2.0)])
    r2_same = rung_key(_inverter(), "R2") == rung_key(swap, "R2")
    r3_diff = rung_key(_inverter(), "R3") != rung_key(swap, "R3")
    print(f"  [{'ok' if r2_same else 'FAIL'}] R2 merges a gate/drain swap")
    print(f"  [{'ok' if r3_diff else 'FAIL'}] R3 separates it once terminals are typed")
    ok &= r2_same and r3_diff

    # 3. the nesting Proposition 1 assumes, on random circuits
    corpus = [_random_netlist(rng) for _ in range(60)]
    res = check_nesting(ladder_keys(corpus))
    print(f"  [{'ok' if res['ok'] else 'FAIL'}] ladder nesting holds on {len(corpus)} random "
          f"circuits ({res['n_violations']} violations)")
    ok &= bool(res["ok"])

    # 4. quantization coarsens monotonically: wider bins cannot split a class
    fine = [rung_key(c, "R2", bin_width=0.10) for c in corpus]
    coarse = [rung_key(c, "R2", bin_width=1.00) for c in corpus]
    pairs = [(i, j) for i in range(len(corpus)) for j in range(i + 1, len(corpus))]
    merged = sum(coarse[i] == coarse[j] and fine[i] != fine[j] for i, j in pairs)
    split = sum(coarse[i] != coarse[j] and fine[i] == fine[j] for i, j in pairs)
    print(f"  [{'ok' if split == 0 else 'FAIL'}] wider quantization only merges classes, "
          f"never splits ({merged} pairs merged, {split} split)")
    ok &= split == 0

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    print("SimBoundary ladder self-test")
    sys.exit(0 if run_self_test() else 1)
