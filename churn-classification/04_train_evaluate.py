import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

X_train = pd.read_csv("x_train.csv")
X_test = pd.read_csv("x_test.csv")
y_train = pd.read_csv("y_train.csv")["Exited"]
y_test = pd.read_csv("y_test.csv")["Exited"]

knn = KNeighborsClassifier().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)
print("Model training selesai.\n")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Confusion Matrix": cm,
        "True Positive (TP)": tp,
        "False Positive (FP)": fp,
        "False Negative (FN)": fn,
        "True Negative (TN)": tn,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }

results = {
    "K-Nearest Neighbors (KNN)": evaluate_model(knn, X_test, y_test),
    "Decision Tree (DT)": evaluate_model(dt, X_test, y_test),
    "Random Forest (RF)": evaluate_model(rf, X_test, y_test),
    "Support Vector Machine (SVM)": evaluate_model(svm, X_test, y_test),
    "Naive Bayes (NB)": evaluate_model(nb, X_test, y_test),
}

rows = []
for model_name, metrics in results.items():
    rows.append(
        {
            "Model": model_name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
        }
    )

summary_df = pd.DataFrame(rows)
print("=== RINGKASAN HASIL EVALUASI ===")
print(summary_df.to_string(index=False))

print("\n=== CONFUSION MATRIX PER MODEL ===")
for model_name, metrics in results.items():
    cm = metrics["Confusion Matrix"]
    print(f"{model_name}:")
    print(f"  TP={metrics['True Positive (TP)']}  FP={metrics['False Positive (FP)']}")
    print(f"  FN={metrics['False Negative (FN)']}  TN={metrics['True Negative (TN)']}")

best = summary_df.loc[summary_df["F1-Score"].idxmax()]
print(f"\nModel terbaik berdasarkan F1-Score: {best['Model']} (F1={best['F1-Score']})")