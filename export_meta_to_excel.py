import json
from pathlib import Path

import pandas as pd

def main():
    rows = []
    base_dir = Path("resultados")
    for dataset_dir in base_dir.iterdir():
        meta_dir = dataset_dir / "meta"
        if not meta_dir.exists():
            continue
        for meta_file in meta_dir.glob("meta_*.json"):
            data = json.loads(meta_file.read_text())
            data.setdefault("dataset", dataset_dir.name)
            if "step" not in data:
                try:
                    tag = data["output_json"].split("_s", 1)[1].split(".json")[0]
                    data["step"] = float(tag) / 1000.0
                except Exception:
                    data["step"] = 0.0
            rows.append(data)

    df = pd.DataFrame(rows)
    output_path = base_dir / "meta_resumen.csv"
    df.to_csv(output_path, index=False)
    print(f"Guardado {output_path} con {len(df)} filas")

if __name__ == "__main__":
    main()
