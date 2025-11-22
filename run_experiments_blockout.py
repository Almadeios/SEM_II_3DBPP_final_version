import itertools
import subprocess
import sys

BUFFERS = [1, 3, 5, 10]
STEPS = [0.01, 0.02, 0.03]

BASE_CMD = [
    sys.executable,
    "upgraded_secuencial.py",
    "--dataset",
    "blockout",
    "--sequence-index",
    "3",
    "--restrict-rotations",
    "--zhao-order",
    "--metaheuristic",
    "grasp",
    "--grasp-iterations",
    "10",
    "--rcl-size",
    "8",
    "--max-passes",
    "2",
    "--random-seed",
    "42",
]


def main():
    for buffer_size, step in itertools.product(BUFFERS, STEPS):
        cmd = BASE_CMD + ["--buffer-size", str(buffer_size), "--step", str(step)]
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
