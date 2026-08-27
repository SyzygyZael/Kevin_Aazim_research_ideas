import json
from pathlib import Path
import pandas as pd

def build_falcon_jsonl(dataset_dir: str, out_file: str):
    dataset_path = Path(dataset_dir)
    records = []

    # Iterate through all topology subfolders containing dataset.csv and graph.json
    for csv_path in dataset_path.glob("*/*/dataset.csv"):
        folder = csv_path.parent
        graph_file = folder / "graph.json"
        
        if not graph_file.exists():
            continue

        # Load topology structure
        with open(graph_file, "r") as f:
            graph_data = json.load(f)

        # Load sizing variations and performance metrics
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            # 1. Map devices with graph terminals and row-specific parameter values
            devices = []
            for dev in graph_data.get("nodes", []):
                kind = str(dev.get("type", dev.get("kind", "unknown"))).lower()
                dev_id = dev.get("id", "")
                
                # Assign terminal roles ('g','d','s' for transistors, 'p','n' for passives)
                nets = dev.get("nets", {})
                
                # Fetch value/sizing from CSV column if present (e.g. W/L or passive value)
                val = float(row[dev_id]) if dev_id in row else None
                
                devices.append({
                    "kind": kind,
                    "value": val,
                    "terminals": nets
                })

            # 2. Extract SPICE metrics (e.g. DC power, gain, noise figure, bandwidth)
            # Adjust column names below to match your dataset.csv headers
            properties = {
                "dc_power_w": float(row["dc_power"]) if "dc_power" in row else float(row.get("power", 0.0)),
                "gain_db": float(row["gain"]) if "gain" in row else float(row.get("gain_db", 0.0)),
            }

            records.append({
                "name": f"{folder.name}_rec_{idx}",
                "devices": devices,
                "properties": properties
            })

    # Write out consolidated JSON Lines file
    out_path = Path(out_file)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully wrote {len(records)} records to {out_file}")

if __name__ == "__main__":
    # Point this to your top-level 'dataset' folder
    build_falcon_jsonl("dataset", "falcon.jsonl")