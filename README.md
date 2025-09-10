In this notebook, we will:

Build ML models (classification & regression) using non-linear algorithms.

Perform hyperparameter tuning using different methods.

Compare results and show how performance changed with each modification.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

Load and prepare the data

# Classification dataset
X_class, y_class = load_breast_cancer(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Regression dataset
X_reg, y_reg = fetch_california_housing(return_X_y=True, as_frame=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
Xc_train = scaler.fit_transform(Xc_train)
Xc_test = scaler.transform(Xc_test)
Xr_train = scaler.fit_transform(Xr_train)
Xr_test = scaler.transform(Xr_test)


Train base non-linear models

# --- Classification ---
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

rf_clf.fit(Xc_train, yc_train)
gb_clf.fit(Xc_train, yc_train)

print("Random Forest Class Acc:", accuracy_score(yc_test, rf_clf.predict(Xc_test)))
print("Gradient Boosting Class Acc:", accuracy_score(yc_test, gb_clf.predict(Xc_test)))

# --- Regression ---
rf_reg = RandomForestRegressor(random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)

rf_reg.fit(Xr_train, yr_train)
gb_reg.fit(Xr_train, yr_train)

print("Random Forest Regr RMSE:", mean_squared_error(yr_test, rf_reg.predict(Xr_test), squared=False))
print("Gradient Boosting Regr RMSE:", mean_squared_error(yr_test, gb_reg.predict(Xr_test), squared=False))


Hyperparameter Tuning

param_grid_clf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_rf_clf = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid_clf, cv=3, scoring="accuracy", n_jobs=-1)
grid_rf_clf.fit(Xc_train, yc_train)

print("Best RF Classifier Params:", grid_rf_clf.best_params_)
print("Best CV Accuracy:", grid_rf_clf.best_score_)


Randomized Search

param_dist_reg = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "learning_rate": np.linspace(0.01, 0.3, 5),
    "subsample": [0.6, 0.8, 1.0]
}

rand_gb_reg = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), 
                                 param_dist_reg, n_iter=10, cv=3, 
                                 scoring="neg_root_mean_squared_error", 
                                 n_jobs=-1, random_state=42)

rand_gb_reg.fit(Xr_train, yr_train)

print("Best GB Regressor Params:", rand_gb_reg.best_params_)
print("Best CV RMSE:", -rand_gb_reg.best_score_)


Compare results

# Classification
print("\n--- Classification Results ---")
print("Base RF:", accuracy_score(yc_test, rf_clf.predict(Xc_test)))
print("Tuned RF:", accuracy_score(yc_test, grid_rf_clf.best_estimator_.predict(Xc_test)))

# Regression
print("\n--- Regression Results ---")
print("Base GB RMSE:", mean_squared_error(yr_test, gb_reg.predict(Xr_test), squared=False))
print("Tuned GB RMSE:", mean_squared_error(yr_test, rand_gb_reg.best_estimator_.predict(Xr_test), squared=False))


Visualize feature importances

# Classification RF feature importance
importances = grid_rf_clf.best_estimator_.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), indices)
plt.xlabel("Feature Importance")
plt.title("Top 10 Features (Classification RF)")
plt.show()

# Regression GB feature importance
importances_reg = rand_gb_reg.best_estimator_.feature_importances_
indices_reg = np.argsort(importances_reg)[-10:]

plt.figure(figsize=(8,5))
plt.barh(range(len(indices_reg)), importances_reg[indices_reg], align="center")
plt.yticks(range(len(indices_reg)), indices_reg)
plt.xlabel("Feature Importance")
plt.title("Top 10 Features (Regression GB)")
plt.show()

