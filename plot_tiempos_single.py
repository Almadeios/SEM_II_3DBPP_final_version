"""
Lee resultados/tiempos_single.csv y genera un grafico por dataset
con BufferSize en el eje X y el tiempo (segundos) en el eje Y,
con una serie por cada step. Guarda el PNG en resultados/tiempos_single.png.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    csv_path = Path("resultados") / "tiempos_single.csv"
    df = pd.read_csv(csv_path)

    datasets = df["dataset"].unique()
    steps = sorted(df["step"].unique(), key=float)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].sort_values(["buffer_size", "step"], key=lambda s: s.astype(float))
        for step in steps:
            sub_step = sub[sub["step"] == step]
            ax.plot(
                sub_step["buffer_size"],
                sub_step["time_s"],
                marker="o",
                label=f"step={step}",
            )
        ax.set_title(ds)
        ax.set_xlabel("Buffer size (k)")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Tiempo (s)")
    axes[-1].legend(title="Step")
    fig.tight_layout()

    out_png = Path("resultados") / "tiempos_single.png"
    fig.savefig(out_png, dpi=150)
    print(f"Grafico guardado en {out_png}")


if __name__ == "__main__":
    main()
