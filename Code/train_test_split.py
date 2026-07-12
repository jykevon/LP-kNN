from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

filename = "final_cut_scaled.csv"


dataframe = pd.read_csv(filename)
classname = dataframe.columns[-1]  # last column is the class column

X, y = dataframe.drop(columns=[classname]), dataframe[classname]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50, stratify=y
)

train_df = X_train.copy()
train_df[classname] = y_train

test_df = X_test.copy()
test_df[classname] = y_test

# --------------------
# Shared label mapping
# --------------------
unique_labels = sorted(dataframe[classname].dropna().unique())
label_to_int = {lab: i for i, lab in enumerate(unique_labels)}

y_train_int = y_train.map(label_to_int)
y_test_int = y_test.map(label_to_int)

# --------------------
# Plot
# --------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# ---- Train ----
sc = axes[0].scatter(
    X_train.iloc[:, 0],
    X_train.iloc[:, 1],
    c=y_train_int,
    cmap="coolwarm",
    s=5,
)
axes[0].set_title("Training Dataset")
axes[0].set_xlabel(X.columns[0])
axes[0].set_ylabel(X.columns[1])
axes[0].set_aspect("equal", adjustable="box")

# ---- Test ----
axes[1].scatter(
    X_test.iloc[:, 0],
    X_test.iloc[:, 1],
    c=y_test_int,
    cmap="coolwarm",
    s=5,
)
axes[1].set_title("Test Dataset")
axes[1].set_xlabel(X.columns[0])
axes[1].set_aspect("equal", adjustable="box")

# ---- Shared legend ----
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

fig.legend(
    handles=handles,
    title=classname,
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
)

plt.tight_layout()
plt.show()

train_df.to_csv(f"{Path(filename).stem}_train.csv", index=False)
test_df.to_csv(f"{Path(filename).stem}_test.csv", index=False)
