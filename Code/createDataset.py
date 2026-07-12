from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SAMPLES = 250
NOISE = 0.1
SCALE = 10
RANDOM_STATE = 100

centers = [(0, 0), (2, 0), (1, 1)]

X, y = make_blobs(
    n_samples=(SAMPLES // 2, SAMPLES // 2, SAMPLES // 2),
    centers=centers,
    cluster_std=0.6,
    random_state=RANDOM_STATE,
)


X = X * SCALE

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
plt.title("Small Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.tight_layout()
plt.show()

df = pd.DataFrame(X, columns=["x1", "x2"])
df["label"] = y
df.to_csv("small_data.csv", index=False)
