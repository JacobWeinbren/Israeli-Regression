import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score
import pyreadstat
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load data from .sav file
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")
logging.info(f"Data loaded with shape {df.shape}")

# Drop non-informative features
non_informative_features = ["agree_1", "citizen‎"]
df.drop(columns=non_informative_features, inplace=True)
logging.info("Dropped non-informative features.")

# Convert all empty strings to NaN
df.replace("", np.nan, inplace=True)

# Drop rows with missing values in key columns
df = df.dropna(subset=["age_group", "sex", "educ", "v144", "v104"])
logging.info("Dropped rows with missing values in key columns.")

# Drop specified features
df.drop(columns=["v104a", "w_panel2", "ID", "F2", "v706"], inplace=True)
logging.info("Dropped features 'v104a' and 'w_panel2'.")

# Convert categorical variables to codes
categorical_features = ["age_group", "sex", "educ", "v144", "v104"]
for feature in categorical_features:
    df[feature] = df[feature].astype("category").cat.codes
logging.info("Converted categorical variables to codes.")

# Define features and target
X = df.drop("v104", axis=1)
Y = df["v104"]

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors="coerce")
logging.info("Ensured all columns are numeric.")

# Identify columns with all NaN values and exclude them from imputation
cols_with_values = X.columns[X.notna().any()]
X_filtered = X[cols_with_values]
logging.info(f"Filtered columns with non-NaN values: {cols_with_values.tolist()}")

# Handle NaN values using an imputer
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_filtered)
logging.info("Applied mean imputation.")

# Create a DataFrame from the imputed data using the correct columns
X_imputed_df = pd.DataFrame(X_imputed, columns=cols_with_values)
logging.info("DataFrame created from imputed data.")

# Continue with scaling and other preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
logging.info("Data scaling applied.")

# Handling imbalanced data
min_class_count = Y.value_counts().min()
smote_neighbors = max(1, min_class_count - 1)  # Ensure at least one neighbor
smote = SMOTE(k_neighbors=smote_neighbors)
X_smote, Y_smote = smote.fit_resample(X_scaled, Y)
logging.info("Applied SMOTE to handle class imbalance.")

# Train preliminary RandomForest to get feature importances
prelim_model = RandomForestClassifier(n_estimators=100, random_state=42)
prelim_model.fit(X_smote, Y_smote)
importances = prelim_model.feature_importances_

# Get the indices of the top 10 features
indices = np.argsort(importances)[::-1][:10]
top_10_features = cols_with_values[indices]
logging.info(f"Top 10 selected features: {top_10_features.tolist()}")

# Select only the top 10 features for further modeling
X_top_10 = X_smote[:, indices]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_top_10, Y_smote, test_size=0.2, stratify=Y_smote, random_state=42
)
logging.info("Data split into train and test sets.")

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", verbose=1)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
logging.info(f"Best model parameters: {grid_search.best_params_}")

# Predict and evaluate
Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
logging.info(f"Model accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print(f"Accuracy of the model: {accuracy:.2f}")

# Calculate R^2 value
r2_value = r2_score(Y_test, Y_pred)
print(f"R^2 value: {r2_value:.2f}")

# Feature importance
feature_importances = best_model.feature_importances_
print("Feature importances:\n", feature_importances)
