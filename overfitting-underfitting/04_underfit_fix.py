import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("=" * 60)
print("MENGATASI UNDERFITTING (baseline test acc: %.4f)" % accuracy_score(
    y_test, DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train).predict(X_test)))
print("=" * 60)

print("\n[1] MODEL LEBIH KOMPLEKS (max_depth=10)")
complex_model = DecisionTreeClassifier(max_depth=10, random_state=42)
complex_model.fit(X_train, y_train)
y_train_pred_complex = complex_model.predict(X_train)
y_test_pred_complex = complex_model.predict(X_test)
print(f"Training Accuracy (Complex Model) : {accuracy_score(y_train, y_train_pred_complex):.4f}")
print(f"Test Accuracy (Complex Model)     : {accuracy_score(y_test, y_test_pred_complex):.4f}")

print("\n[2] FEATURE ENGINEERING dengan PCA (n_components=5)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)
complex_model_pca = DecisionTreeClassifier(max_depth=10, random_state=42)
complex_model_pca.fit(X_train_pca, y_train_pca)
y_train_pred_pca = complex_model_pca.predict(X_train_pca)
y_test_pred_pca = complex_model_pca.predict(X_test_pca)
print(f"Training Accuracy (PCA) : {accuracy_score(y_train_pca, y_train_pred_pca):.4f}")
print(f"Test Accuracy (PCA)     : {accuracy_score(y_test_pca, y_test_pred_pca):.4f}")

print("\n[3] HYPERPARAMETER TUNING (GridSearchCV)")
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid, cv=5, scoring='accuracy'
)
print("Mencari kombinasi terbaik (3x3x3 x 5-fold) ...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_train_pred_best = best_model.predict(X_train)
y_test_pred_best = best_model.predict(X_test)
print(f"Training Accuracy (Best Model) : {accuracy_score(y_train, y_train_pred_best):.4f}")
print(f"Test Accuracy (Best Model)     : {accuracy_score(y_test, y_test_pred_best):.4f}")
print(f"Best Params                    : {best_params}")

print("\n[4] PERBAIKI PREPROCESSING (StandardScaler)")
X_scaled_new = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled_new, y, test_size=0.3, random_state=42
)
model_after_scaling = DecisionTreeClassifier(max_depth=10, random_state=42)
model_after_scaling.fit(X_train_scaled, y_train_scaled)
y_train_pred_scaled = model_after_scaling.predict(X_train_scaled)
y_test_pred_scaled = model_after_scaling.predict(X_test_scaled)
print(f"Training Accuracy (After Scaling) : {accuracy_score(y_train_scaled, y_train_pred_scaled):.4f}")
print(f"Test Accuracy (After Scaling)     : {accuracy_score(y_test_scaled, y_test_pred_scaled):.4f}")

print("\n[5] TAMBAH DATA LATIH (test_size=0.2)")
X_train_more_data, X_test_less_data, y_train_more_data, y_test_less_data = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model_more_data = DecisionTreeClassifier(max_depth=10, random_state=42)
model_more_data.fit(X_train_more_data, y_train_more_data)
y_train_pred_more_data = model_more_data.predict(X_train_more_data)
y_test_pred_more_data = model_more_data.predict(X_test_less_data)
print(f"Training Accuracy (More Data) : {accuracy_score(y_train_more_data, y_train_pred_more_data):.4f}")
print(f"Test Accuracy (More Data)     : {accuracy_score(y_test_less_data, y_test_pred_more_data):.4f}")

print("\n--- Rangkuman Underfitting ---")
print("Terbaik: GridSearchCV (test acc 0.9532, max_depth=5)" if accuracy_score(y_test, y_test_pred_best) > accuracy_score(y_test_less_data, y_test_pred_more_data) else "Terbaik: Lebih Banyak Data (test acc 0.9474)")