import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

filename = "results"
df = pd.read_csv(f"{filename}.csv")

pivot_train = df.pivot(index="k", columns="R", values="train_accuracy")
pivot_test = df.pivot(index="k", columns="R", values="test_accuracy")

vmin = min(pivot_train.min().min(), pivot_test.min().min())
vmax = max(pivot_train.max().max(), pivot_test.max().max())

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, pivot, title in zip(
    axes,
    [pivot_train, pivot_test],
    ["Train Accuracy", "Test Accuracy"],
):
    im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xlabel("Propagation Radius $R$", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.4g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

axes[0].set_ylabel("Number of Neighbors $k$", fontsize=12)

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="3%", pad=0.25)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Accuracy", fontsize=12)

fig.suptitle(
    "Accuracy Heatmaps for Lattice-Propagated $k$-Nearest Neighbors",
    fontsize=15,
    y=1,
)

plt.tight_layout()
plt.savefig(f"{filename}_train_test.png", dpi=300, bbox_inches="tight")
plt.savefig("accuracy_grid.png")
