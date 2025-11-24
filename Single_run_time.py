import csv
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ["general", "kitchen", "blockout"]
BUFFER_SIZES = [1, 3, 5, 10] 
STEPS = ["0.01", "0.02", "0.03"]

DATASET_FLAGS = {
    "general": ["--sequence-index", "3", "--restrict-rotations", "--zhao-order"],
    "kitchen": ["--sequence-index", "3"],
    "blockout": ["--sequence-index", "3", "--restrict-rotations", "--zhao-order"],
}

COMMON_ARGS = [
    "--metaheuristic", "grasp",
    "--grasp-iterations", "1",
    "--rcl-size", "8",
    "--max-passes", "2",
    "--random-seed", "42",
]

def main():
    out_path = Path("resultados") / "tiempos_single.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    for ds in DATASETS:
        for buf in BUFFER_SIZES:
            for step in STEPS:
                cmd = [
                    sys.executable, "upgraded_secuencial.py",
                    "--dataset", ds,
                    "--buffer-size", str(buf),
                    "--step", step,
                    *DATASET_FLAGS.get(ds, []),
                    *COMMON_ARGS,
                ]
                print(">>>", " ".join(cmd))
                t0 = time.perf_counter()
                subprocess.run(cmd, check=True)
                elapsed = time.perf_counter() - t0
                print(f"{ds} | K={buf} | Step={step} | Time={elapsed:.2f} s\n")

                with out_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["dataset", "buffer_size", "step", "time_s"])
                        write_header = False
                    writer.writerow([ds, buf, step, f"{elapsed:.4f}"])
    print(f"Tiempos guardados en {out_path}")

if __name__ == "__main__":
    main()
