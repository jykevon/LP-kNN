import numpy as np
import matplotlib.pyplot as plt

x, y = 2.7, 1.3
x_round, y_round = round(x), round(y)

xmin, xmax = 0, 5
ymin, ymax = 0, 5

plt.figure(figsize=(5, 5))

for i in range(xmin, xmax + 1):
    plt.axvline(i, color="lightgray", linewidth=0.8)
for j in range(ymin, ymax + 1):
    plt.axhline(j, color="lightgray", linewidth=0.8)

plt.scatter(x, y, color="red", s=80, zorder=3)
plt.text(x + 0.05, y + 0.05, r"$(x, y)$", color="red")

plt.scatter(x_round, y_round, color="blue", s=80, zorder=3)
plt.text(
    x_round + 0.05,
    y_round + 0.05,
    r"$(\lfloor x \rceil,\; \lfloor y \rceil)$",
    color="blue",
)

plt.arrow(
    x,
    y,
    x_round - x,
    y_round - y,
    length_includes_head=True,
    head_width=0.08,
    head_length=0.12,
    fc="black",
    ec="black",
    linewidth=1.5,
    zorder=2,
)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Componentwise Rounding to the Integer Lattice")
plt.tight_layout()
plt.savefig("lattice_construct.png")
