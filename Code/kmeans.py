import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

def show_clusters(X, kmeans, y, k):
    plt.figure(figsize=(8,6))

    # Plot points colored by **true class**
    for class_id in np.unique(y):
        points = X[y == class_id]
        plt.scatter(points[:,0], points[:,1], label=f"True Class {class_id}", alpha=0.6)

    # Overlay cluster assignments (optional: outlines)
    for cluster_id in range(k):
        points = X[kmeans.labels_ == cluster_id]
        plt.scatter(points[:,0], points[:,1], facecolors='none', edgecolors='black', s=60, linewidths=1)

    # Plot cluster centers
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                color='red', marker='X', s=100, label='Cluster Centers')

    # Annotate clusters with majority class
    for i, center in enumerate(kmeans.cluster_centers_):
        plt.text(center[0], center[1], f"{cluster_labels[i]}", fontsize=12, horizontalalignment='center', verticalalignment='center', color='black')

    plt.title("K-Means Clustering with True Labels")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def log_confusion_matrix(actual_values, predicted_values):
    cm = confusion_matrix(actual_values, predicted_values)
    labels = sorted(set(actual_values) | set(predicted_values))

    labels = [str(l) for l in labels]

    max_label_width = max(len(l) for l in labels)
    max_value_width = max(len(str(v)) for row in cm for v in row)
    col_width = max(max_label_width, max_value_width) + 2

    print("\nConfusion Matrix\n")

    header = "Pred \\ Actual".ljust(col_width) + "".join(
        l.rjust(col_width) for l in labels
    )
    print(header)

    for label, row in zip(labels, cm):
        row_str = label.ljust(col_width) + "".join(str(v).rjust(col_width) for v in row)
        print(row_str)

    return cm.tolist()

def logMetrics(confusionMatrix):
    """
    rows = actual, cols = predicted
    """

    attributes = len(confusionMatrix)
    precisions = []
    recalls = []
    f1s = []

    for i in range(attributes):
        tp = confusionMatrix[i][i]

        # Precision_i = TP / (TP + FP) = TP / (sum of column i)
        col_sum = 0
        for j in range(attributes):
            col_sum += confusionMatrix[j][i]

        if col_sum == 0:
            recall = 0.0
        else:
            recall = tp / col_sum
        recalls.append(recall)

        # Recall_i = TP / (TP + FN) = TP / (sum of row i)
        row_sum = 0
        for j in range(attributes):
            row_sum += confusionMatrix[i][j]

        if row_sum == 0:
            precision = 0.0
        else:
            precision = tp / row_sum
        precisions.append(precision)

        # F1_i = 2PR/(P+R)
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)

    trace = sum(confusionMatrix[i][i] for i in range(attributes))
    matrixSum = sum(sum(confusionMatrix[i]) for i in range(attributes))
    accuracy = trace / matrixSum if matrixSum != 0 else 0.0

    print("Metrics:")
    print("Precision:")
    for i in range(len(precisions)):
        print(f"\tCol {i}: {precisions[i]}")
    print("Recall:")
    for i in range(len(recalls)):
        print(f"\tRow {i}: {recalls[i]}")
    print("F1 Score:")
    for i in range(len(f1s)):
        print(f"\tClass {i}: {f1s[i]}")
    print(f"Accuracy: {accuracy}")

    return {
        "accuracy": accuracy,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1s,
    }

def load_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    X = df[["longitude", "latitude"]].values
    y = df["alignment"].values

    return X, y

k_values = range(1, 101, 10)
inertia_values = []

for k in k_values:
    X_train, y_train = load_from_csv("final_cut_scaled_train3.csv")
    X_test, y_test = load_from_csv("final_cut_scaled_test3.csv")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)

    cluster_labels = {}
    for cluster_id in range(k):
        points_in_cluster = y_train[kmeans.labels_ == cluster_id]

        majority_label = Counter(points_in_cluster).most_common(1)[0][0]
        cluster_labels[cluster_id] = majority_label

    #show_clusters(X_train, kmeans, y_train, k)

    test_clusters = kmeans.predict(X_test)

    y_pred = np.array([cluster_labels[cluster_id] for cluster_id in test_clusters])

    print(k)

    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Democrat', "Actual Republican"], columns=['Pred Democrat', "Pred Republican"])
    print('Confusion Matrix\n', cm_df, '\n')
    print('accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('f1: ' + str(f1_score(y_test, y_pred, average="macro")))
    print('recall: ' + str(recall_score(y_test, y_pred, average="macro")))
    print('precision: ' + str(precision_score(y_test, y_pred, average="macro")))
    cm = log_confusion_matrix(y_pred, list(y_test))
    metrics = logMetrics(cm)


plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_values, 'o-', color='blue')
plt.xlabel("Number of clusters k")
plt.ylabel("WCSS (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()
