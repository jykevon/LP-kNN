import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

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

def load_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    X = df[["longitude", "latitude"]].values
    y = df["alignment"].values

    return X, y

X_train, y_train = load_from_csv("final_cut_scaled_train3.csv")
X_test, y_test = load_from_csv("final_cut_scaled_test3.csv")

#print(X_train)
#print(y_train)

tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=42)

tree.fit(X_train, y_train)

'''
feature_names = ["latitude", "longitude"]
tree_rules = export_text(tree, feature_names=feature_names)
print(tree_rules)
'''

y_pred = tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Democrat', "Actual Republican"], columns=['Pred Democrat', "Pred Republican"])
print('Confusion Matrix\n', cm_df, '\n')
print('accuracy: ' + str(accuracy_score(y_test, y_pred)))
print('f1: ' + str(f1_score(y_test, y_pred, average="macro")))
print('recall: ' + str(recall_score(y_test, y_pred, average="macro")))
print('precision: ' + str(precision_score(y_test, y_pred, average="macro")))