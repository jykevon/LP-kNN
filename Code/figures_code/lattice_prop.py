import matplotlib.pyplot as plt
import numpy as np


R = 3
k = 5
center = np.array([0, 0])

points = np.array(
    [[-2, -1], [-1, 1], [0, 2], [1, 1], [2, 0], [3, 1], [-3, 0], [1, -2], [2, -2]]
)

d2 = np.sum((points - center) ** 2, axis=1)

inside = d2 <= R**2

nearest_idx = np.argsort(d2[inside])[:k]
knn_points = points[inside][nearest_idx]

plt.figure(figsize=(5, 5))

plt.scatter(points[:, 0], points[:, 1], s=60, label="Labeled lattice points")

plt.scatter(points[inside, 0], points[inside, 1], s=60, label="Within radius $R$")

plt.scatter(
    knn_points[:, 0], knn_points[:, 1], s=120, marker="o", label="$k$ nearest neighbors"
)

plt.scatter(center[0], center[1], s=150, marker="x", label="Unlabeled point $p$")

theta = np.linspace(0, 2 * np.pi, 400)
plt.plot(R * np.cos(theta), R * np.sin(theta), linestyle="--", label="Radius $R$")

plt.gca().set_aspect("equal")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Lattice Propagation for a Single Point")
plt.legend(loc="upper center", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
plt.grid(True)
plt.tight_layout()

plt.savefig("lattice_prop.png")
