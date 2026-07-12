import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from confusionMatrix import logMetrics, log_confusion_matrix
import math

DEBUG = True


class LP_kNN:
    def __init__(self, k: int, R: float, attribute_min_max=None):
        # attribute_min_max must be [[min,max], ...]
        # lattice_scales and attribute_min_max size mismatch

        self.k = int(k)
        self.R = float(R)
        self.R_squared = R**2

        if attribute_min_max is not None:
            self.min_max = np.array(attribute_min_max)

            self.attr_min = self.min_max[:, 0]
            self.attr_max = self.min_max[:, 1]
        else:
            # Will be inferred later in fit()
            self.min_max = None
            self.attr_min = None
            self.attr_max = None

        self.lattice = None

    def init_lattice(self, X: np.DataFrame):

        lattice = {}

        for a1_val in range(self.attr_min[0], self.attr_max[0]):
            for a2_val in range(self.attr_min[1], self.attr_max[1]):
                lattice[(a1_val, a2_val)] = None

        return lattice

    def clip_to_lattice(self, data: pd.DataFrame, lattice: dict):

        for row in data.itertuples(index=False):
            x, y, c = row

            x_rounded = round(x)
            y_rounded = round(y)

            if lattice[(x_rounded, y_rounded)] == None:
                lattice[(x_rounded, y_rounded)] = [c]
            else:
                lattice[(x_rounded, y_rounded)].append(c)

        for key in lattice.keys():
            x, y = key
            if lattice[(x, y)] != None:
                lattice[(x, y)] = Counter(lattice[(x, y)]).most_common(1)[0][0]

    def get_cell(self, x: int, y: int):
        return self.lattice.get((x, y), None)

    def dist(self, a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(len(a)))

    def get_neighboring_points(self, point):
        x, y = point
        R = int(self.R)
        R2 = self.R_squared

        neighbors = []
        for i in range(x - R, x + R + 1):
            for j in range(y - R, y + R + 1):
                if (i - x) ** 2 + (j - y) ** 2 <= R2:
                    neighbors.append((i, j))

        return neighbors

    def get_all_close_points(self, point, all_filled_points, lattice: dict):
        points = []
        for filled_point in all_filled_points:
            x, y = filled_point
            if self.dist(point, filled_point) <= self.R_squared:
                points.append((x, y))
        return points

    def KNN(self, all_close_points: list, lattice: dict, center=None):
        if len(all_close_points) < self.k:
            return None

        if center is None:
            # fall back to unweighted majority
            return Counter([lattice[p] for p in all_close_points]).most_common(1)[0][0]

        weights = {}
        for p in all_close_points:
            label = lattice[p]
            d2 = self.dist(center, p)
            if d2 == 0:
                w = 1.0
            else:
                w = 1.0 / d2
            weights[label] = weights.get(label, 0.0) + w

        # argmax over weighted votes
        return max(weights.items(), key=lambda kv: kv[1])[0]

    def fill_lattice(self, lattice: dict):

        all_none_points = set({key for key in lattice.keys() if lattice[key] == None})
        all_filled_points = set({key for key in lattice.keys() if lattice[key] != None})

        while len(all_none_points) != 0:
            print(f"{len(all_none_points)} more to fill")
            self.log_lattice(title=f"{len(all_none_points)}_points_left", save=True)

            none_to_replace = []

            for none_point in all_none_points:

                x, y = none_point

                if (
                    len(all_filled_points) > 4 * self.R_squared
                ):  # 4*R^2 is the specific big O for get_neighboring points
                    neighbors = set(self.get_neighboring_points(none_point))
                    all_close_points = neighbors.intersection(all_filled_points)
                else:
                    all_close_points = self.get_all_close_points(
                        none_point, all_filled_points, lattice
                    )
                label = self.KNN(
                    all_close_points,
                    lattice,
                )

                if label != None:
                    none_to_replace.append((none_point, label))

            for package in none_to_replace:
                point, label = package
                lattice[point] = label
                all_none_points.remove(point)
                all_filled_points.add(point)
        self.log_lattice(title=f"finished", save=True)

    def log_lattice(self, title="Lattice Classification", save=False, buffer=5):
        if DEBUG:
            xs, ys, labels = [], [], []

            for (x, y), label in self.lattice.items():
                if label is not None:
                    xs.append(x)
                    ys.append(y)
                    labels.append(label)

            if not labels:
                print("Lattice is empty.")
                return

            unique_labels = sorted(set(labels))
            label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
            cs = [label_to_int[lab] for lab in labels]

            fig, ax = plt.subplots(figsize=(6, 6))
            scatter = ax.scatter(xs, ys, c=cs, cmap="coolwarm", s=1)

            ax.set_title(title)

            # --- axis bounds from attribute min/max + buffer ---
            if self.attr_min is not None and self.attr_max is not None:
                x_min = self.attr_min[0] - buffer
                x_max = self.attr_max[0] + buffer
                y_min = self.attr_min[1] - buffer
                y_max = self.attr_max[1] + buffer

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

            # keep equal scaling without fighting the limits
            ax.set_aspect("equal", adjustable="box")

            # Legend
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=scatter.cmap(scatter.norm(label_to_int[lab])),
                    label=str(lab),
                    markersize=6,
                )
                for lab in unique_labels
            ]

            ax.legend(
                handles=handles,
                title="Class Label",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

            fig.tight_layout()
            if not save:
                plt.show()
            else:
                fig.savefig(f"DEBUG/{title}", dpi=300, bbox_inches="tight")
                plt.close(fig)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if X.ndim != 2:
            raise ValueError(
                "For the scope of this project, we will work with 2D but this is possible for 3D"
            )

        if self.min_max is None:  # finding the bounds automatically
            data_min = X.min(axis=0).values
            data_max = X.max(axis=0).values
            span = data_max - data_min

            buffer_frac = 0.05
            pad = buffer_frac * span

            pad[span == 0] = 1.0

            attr_min = np.floor(data_min - pad).astype(int)
            attr_max = np.ceil(data_max + pad).astype(int)

            self.attr_min = attr_min
            self.attr_max = attr_max
            self.min_max = np.stack([self.attr_min, self.attr_max], axis=1)

        self.train = X.copy()
        self.train["class"] = y.copy()
        self.X_train = X
        self.y_train = y
        self.lattice = self.init_lattice(X)
        print("lattice initialized")
        self.clip_to_lattice(self.train, self.lattice)
        print("lattice clipped")
        print("starting to fill lattice")

        self.fill_lattice(self.lattice)
        print("lattice filled")

    def predict(self, X: pd.DataFrame):

        results = []
        for row in X.itertuples(index=False):
            x, y = row
            x_round = round(x)
            y_round = round(y)
            results.append(self.lattice[(x_round, y_round)])

        return results


if __name__ == "__main__":
    train_df = pd.read_csv("final_cut_scaled_train.csv")
    test_df = pd.read_csv("final_cut_scaled_test.csv")
    classname = train_df.columns[-1]
    X_train = train_df.drop(columns=[classname])
    y_train = train_df[classname]
    X_test = test_df.drop(columns=[classname])
    y_test = test_df[classname]

    results = {"k": [], "R": [], "test_accuracy": [], "train_accuracy": []}

    for k_val in range(5, 6):
        for R_val in range(10, 11):
            if k_val < math.floor(math.pi * R_val**2) / 4:
                print(f"Working on k={k_val}, R={R_val}")

                model = LP_kNN(k=k_val, R=R_val)  # america

                model.fit(X_train, y_train)

                # TRAIN
                predictions = model.predict(X_train)

                train_confusion_matrix = log_confusion_matrix(
                    predictions, list(y_train)
                )
                train_metrics = logMetrics(train_confusion_matrix)

                # TEST
                predictions = model.predict(X_test)

                test_confusion_matrix = log_confusion_matrix(predictions, list(y_test))
                test_metrics = logMetrics(test_confusion_matrix)

                results["k"].append(k_val)
                results["R"].append(R_val)
                results["train_accuracy"].append(train_metrics["accuracy"])
                results["test_accuracy"].append(test_metrics["accuracy"])

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")
