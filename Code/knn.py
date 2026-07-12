from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

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
            precision = 0.0
        else:
            precision = tp / col_sum
        precisions.append(precision)

        # Recall_i = TP / (TP + FN) = TP / (sum of row i)
        row_sum = 0
        for j in range(attributes):
            row_sum += confusionMatrix[i][j]

        if row_sum == 0:
            recall = 0.0
        else:
            recall = tp / row_sum
        recalls.append(recall)

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

df = pd.read_csv('final_cut_scaled_train.csv')
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]
newdf = pd.read_csv('final_cut_scaled_test.csv')
X_test = newdf.iloc[:, :-1]
y_test = newdf.iloc[:, -1]
k_vals = []
acc = []
with open('metrics.txt', 'w') as f:
    f.write('k, accuracy\n')
for i in range(20):
    k = i + 1
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    confusionMatrix = log_confusion_matrix(y_test, y_pred)
    metrics = logMetrics(confusionMatrix)
    k_vals.append(k)                                                        
    acc.append(accuracy_score(y_test, y_pred))

    with open('metrics.txt', 'a') as f:
        f.write(str(k) + ', ' + str(accuracy_score(y_test, y_pred)) + '\n')
        f.write(str(log_confusion_matrix(y_test, y_pred)))
        f.write(str(logMetrics(confusionMatrix)))

maxAcc = max((i, v) for v, i in enumerate(acc))[1]

fig, ax = plt.subplots()
ax.plot(k_vals, acc)
ax.legend()
ax.set_xlabel("k")
ax.set_ylabel("Accuracy")
ax.set_title("k vs. Accuracy")
ax.plot(k_vals[maxAcc], acc[maxAcc], 'o', label='Best k')
ax.legend()
plt.show()
