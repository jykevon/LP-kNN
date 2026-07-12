import matplotlib.pyplot as plt
import pandas as pd

SCALE = 3
INFILE = "final_cut.csv"
OUTFILE = "final_cut_scaled.csv"

df = pd.read_csv(INFILE)
classname = df.columns[-1]

X = df.drop(columns=[classname]).copy()
y = df[classname].copy()

X_scaled = X * SCALE

print(X_scaled)

unique_labels = sorted(y.dropna().unique())
label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
y_int = y.map(label_to_int)

plt.figure(figsize=(6, 6))
sc = plt.scatter(
    X_scaled.iloc[:, 0], X_scaled.iloc[:, 1], c=y_int, cmap="coolwarm", s=5
)

plt.title("Scaled Dataset")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.axis("equal")

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=sc.cmap(sc.norm(label_to_int[lab])),
        label=str(lab),
        markersize=6,
    )
    for lab in unique_labels
]

plt.legend(handles=handles, title=classname, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

out = X_scaled.copy()
out[classname] = y
out.to_csv(OUTFILE, index=False)

print(f"Saved: {OUTFILE}")
print("Label mapping:", label_to_int)
