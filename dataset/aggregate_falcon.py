#!/usr/bin/env python3
"""Convert the checked-out FALCON-style folders into SimBoundary JSONL.

The raw dataset is organized as:

    dataset/<family>/<topology>/{netlist,dataset.csv,values.yaml,graph.json}

For SimBoundary we need one JSON object per simulated circuit:

    {"name": str,
     "devices": [{"kind": str, "value": float | null,
                  "terminals": {role: net_name, ...}}, ...],
     "properties": {metric_name: float | null, ...}}

This script parses the human-readable netlist templates and fills parameter
values from each row of dataset.csv, falling back to values.yaml for fixed
parameters. Missing expressions fail loudly so we do not create fake data.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DATASET_ROOT = Path(__file__).resolve().parent

PRIMARY_VALUE_ATTR = {
    "nmos": "w",
    "pmos": "w",
    "resistor": "r",
    "capacitor": "c",
    "inductor": "l",
    "vsource": "dc",
    "isource": "dc",
    "port": "r",
    "balun": "rin",
}

TERMINAL_ROLES = {
    "nmos": ("d", "g", "s", "b"),
    "pmos": ("d", "g", "s", "b"),
    "resistor": ("p", "n"),
    "capacitor": ("p", "n"),
    "inductor": ("p", "n"),
    "vsource": ("p", "n"),
    "isource": ("p", "n"),
    "port": ("p", "n"),
    "balun": ("in", "out_p", "out_n"),
}

METRIC_MAP = {
    "Bandwidth": "bandwidth_hz",
    "DCPowerConsumption": "dc_power_w",
    "S11": "s11_db",
    "S22": "s22_db",
    "NoiseFigure": "noise_figure_db",
    "PowerGain": "power_gain_db",
    "ConversionGain": "conversion_gain_db",
    "VoltageSwing": "voltage_swing_v",
    "DrainEfficiency": "drain_efficiency",
    "PAE": "pae",
    "PSAT": "psat",
    "VoltageGain": "voltage_gain_db",
    "OscillationFrequency": "oscillation_frequency_hz",
    "TuningRange": "tuning_range_hz",
    "OutputPower": "output_power_dbm",
    "PhaseNoise": "phase_noise_dbc_per_hz",
}

ALL_METRICS = sorted(set(METRIC_MAP.values()))

UNIT_SCALE = {
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "k": 1e3,
    "M": 1e-3,
    "m": 1e-3,
    "U": 1e-6,
    "u": 1e-6,
    "N": 1e-9,
    "n": 1e-9,
    "P": 1e-12,
    "p": 1e-12,
    "F": 1e-15,
    "f": 1e-15,
}

NUMBER_WITH_UNIT = re.compile(
    r"(?<![A-Za-z0-9_])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(MEG|[TGMKkmunpfa-zA-Z])\b"
)
ATTR_RE = re.compile(r"(\w+)\s*=")


@dataclass(frozen=True)
class ComponentTemplate:
    name: str
    kind: str
    terminals: Dict[str, str]
    value_expr: Optional[str]


def parse_number(text: str) -> float:
    """Parse numbers with common SPICE suffixes, e.g. 45n, 2.5K, 800m."""
    s = str(text).strip()
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(MEG|[A-Za-z])?", s)
    if not match:
        raise ValueError(f"not a numeric literal: {text!r}")
    base = float(match.group(1))
    suffix = match.group(2)
    return base * UNIT_SCALE.get(suffix, 1.0)


def parse_values_yaml(path: Path) -> Dict[str, float]:
    """Tiny YAML reader for the simple key: value files in this dataset."""
    values: Dict[str, float] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val:
            values[key.strip()] = parse_number(val)
    return values


def join_continuations(lines: Iterable[str]) -> List[str]:
    joined: List[str] = []
    current = ""
    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("//"):
            continue
        current = f"{current} {line.strip()}".strip() if current else line.strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        joined.append(current)
        current = ""
    if current:
        joined.append(current)
    return joined


def clean_net_name(name: str) -> str:
    return name.replace(r"\+", "+").replace(r"\-", "-")


def extract_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    matches = list(ATTR_RE.finditer(attr_text))
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(attr_text)
        attrs[key] = attr_text[start:stop].strip()
    return attrs


def roles_for(kind: str, nets: Sequence[str]) -> Dict[str, str]:
    roles = TERMINAL_ROLES.get(kind, tuple(f"t{i}" for i in range(len(nets))))
    if len(nets) > len(roles):
        roles = tuple(list(roles) + [f"t{i}" for i in range(len(roles), len(nets))])
    return {roles[i]: clean_net_name(nets[i]) for i in range(len(nets))}


def parse_netlist(path: Path) -> List[ComponentTemplate]:
    templates: List[ComponentTemplate] = []
    for line in join_continuations(path.read_text().splitlines()):
        match = re.match(r"^(\S+)\s+\(([^)]*)\)\s+(\S+)\s*(.*)$", line)
        if not match:
            continue
        name, net_text, kind, attr_text = match.groups()
        kind = kind.lower()
        attrs = extract_attrs(attr_text)
        value_attr = PRIMARY_VALUE_ATTR.get(kind)
        templates.append(ComponentTemplate(
            name=name,
            kind=kind,
            terminals=roles_for(kind, net_text.split()),
            value_expr=attrs.get(value_attr) if value_attr else None,
        ))
    if not templates:
        raise ValueError(f"no components parsed from {path}")
    return templates


class ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, names: Dict[str, float]):
        self.names = names

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported literal {node.value!r}")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.names:
            raise ValueError(f"unknown parameter {node.id!r}")
        return float(self.names[node.id])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("unsupported binary operator")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
            raise ValueError("only sqrt(...) calls are supported")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("sqrt expects exactly one positional argument")
        return math.sqrt(self.visit(node.args[0]))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported expression node {type(node).__name__}")


def normalize_expr(expr: str) -> str:
    expr = expr.strip().strip('"')
    return NUMBER_WITH_UNIT.sub(lambda m: str(float(m.group(1)) * UNIT_SCALE.get(m.group(2), 1.0)), expr)


def evaluate(expr: Optional[str], params: Dict[str, float]) -> Optional[float]:
    if expr is None:
        return None
    normalized = normalize_expr(expr)
    return float(ExpressionEvaluator(params).visit(ast.parse(normalized, mode="eval")))


def row_params(row: Dict[str, str], fixed: Dict[str, float]) -> Dict[str, float]:
    params = dict(fixed)
    for key, value in row.items():
        if value is None or value == "":
            continue
        try:
            params[key] = parse_number(value)
        except ValueError:
            pass
    return params


def properties_from_row(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    props: Dict[str, Optional[float]] = {name: None for name in ALL_METRICS}
    for source_name, value in row.items():
        if source_name in METRIC_MAP:
            props[METRIC_MAP[source_name]] = parse_number(value)
    return props


def build_devices(templates: Sequence[ComponentTemplate], params: Dict[str, float]) -> List[dict]:
    devices = []
    for tpl in templates:
        devices.append({
            "kind": tpl.kind,
            "value": evaluate(tpl.value_expr, params),
            "terminals": tpl.terminals,
        })
    return devices


def conversion_records(topology_dir: Path, limit: Optional[int] = None) -> Iterable[dict]:
    templates = parse_netlist(topology_dir / "netlist")
    fixed = parse_values_yaml(topology_dir / "values.yaml")
    family = topology_dir.parent.name
    topology = topology_dir.name
    with (topology_dir / "dataset.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{topology_dir/'dataset.csv'} has no header")
        metric_columns = set(METRIC_MAP)
        netlist_param_names = set()
        for tpl in templates:
            if tpl.value_expr:
                netlist_param_names.update(re.findall(r"\b[A-Za-z_]\w*\b", tpl.value_expr))
        unknown = sorted(
            col for col in reader.fieldnames
            if col not in metric_columns and col not in netlist_param_names and col not in fixed
        )
        if unknown:
            print(f"warning: {family}/{topology} has CSV columns not used as metrics or primary values: {unknown}")

        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            params = row_params(row, fixed)
            yield {
                "name": f"{family}_{topology}_rec_{idx}",
                "devices": build_devices(templates, params),
                "properties": properties_from_row(row),
            }


def topology_dirs(dataset_dir: Path, only: Optional[str] = None) -> List[Path]:
    dirs = sorted(p.parent for p in dataset_dir.glob("*/*/dataset.csv") if (p.parent / "netlist").exists())
    if only:
        dirs = [p for p in dirs if p.name == only or f"{p.parent.name}/{p.name}" == only]
    if not dirs:
        raise FileNotFoundError(f"no topology folders found under {dataset_dir}")
    return dirs


def build_falcon_jsonl(
    dataset_dir: str | Path = DATASET_ROOT,
    out_file: str | Path = DATASET_ROOT / "falcon_converted.jsonl",
    only: Optional[str] = None,
    limit_per_topology: Optional[int] = None,
) -> int:
    dataset_path = Path(dataset_dir)
    out_path = Path(out_file)
    count = 0
    with out_path.open("w") as out:
        for folder in topology_dirs(dataset_path, only=only):
            before = count
            for record in conversion_records(folder, limit=limit_per_topology):
                out.write(json.dumps(record) + "\n")
                count += 1
            print(f"{folder.parent.name}/{folder.name}: wrote {count - before} records")
    print(f"wrote {count} total records to {out_path}")
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert FALCON folders to SimBoundary JSONL.")
    parser.add_argument("--dataset-dir", default=str(DATASET_ROOT))
    parser.add_argument("--out", default=str(DATASET_ROOT / "falcon_converted.jsonl"))
    parser.add_argument("--only", help="convert one topology, e.g. CGLNA or LNA/CGLNA")
    parser.add_argument("--limit-per-topology", type=int, help="debug limit for each topology")
    args = parser.parse_args(argv)

    build_falcon_jsonl(
        dataset_dir=args.dataset_dir,
        out_file=args.out,
        only=args.only,
        limit_per_topology=args.limit_per_topology,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
