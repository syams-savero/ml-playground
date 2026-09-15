import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("STUDI KASUS 2: UNDERFITTING")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print(f"\nDataset : Breast Cancer ({X.shape[0]} baris) | Fitur: {X.shape[1]}")
print(f"Label   : 0 = Malignant ({sum(y==0)}) | 1 = Benign ({sum(y==1)})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Split 70/30 -> train: {X_train.shape[0]} | test: {X_test.shape[0]}")

model_underfit = DecisionTreeClassifier(max_depth=1, random_state=42)
print("\nTrain DecisionTreeClassifier(max_depth=1) ...")
model_underfit.fit(X_train, y_train)

y_train_pred_underfit = model_underfit.predict(X_train)
y_test_pred_underfit = model_underfit.predict(X_test)

train_acc_underfit = accuracy_score(y_train, y_train_pred_underfit)
test_acc_underfit = accuracy_score(y_test, y_test_pred_underfit)

print("\n--- Deteksi Underfitting ---")
print(f"Underfit Model Training Accuracy : {train_acc_underfit:.4f}")
print(f"Underfit Model Test Accuracy     : {test_acc_underfit:.4f}")
print("\nModel terlalu sederhana (max_depth=1) -> tidak bisa")
print("menangkap pola -> akurasi rendah di KEDUA data.")

train_sizes, train_scores, test_scores = learning_curve(
    model_underfit, X_train, y_train, cv=5,
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

print("\n--- Learning Curve ---")
print(f"Training score tetap ~ {train_scores_mean[-1]:.4f}")
print(f"Validation score tetap ~ {test_scores_mean[-1]:.4f}")
print("=> kedua kurva tidak meningkat dengan tambahan data -> underfitting")

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('Training Set Size')
plt.title('Learning Curve (Underfitting)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_underfit.png")
print("Plot disimpan: learning_curve_underfit.png")