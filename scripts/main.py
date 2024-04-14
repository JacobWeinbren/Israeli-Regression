import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
)
from sklearn.pipeline import Pipeline
import pyreadstat
import logging
import joblib

# Configure logging to monitor progress and debug issues
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load dataset from a .sav file using pyreadstat
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")
logging.info(f"Data loaded with shape {df.shape}")

# Specify the features to be used in the model
selected_features = ["age_group", "sex", "v144", "v712", "v131", "v143_code"]

# Impute missing values using the mean strategy for numerical stability
imputer = SimpleImputer(strategy="mean")
df[selected_features] = imputer.fit_transform(df[selected_features])
# Create polynomial features to explore potential interactions and non-linear relationships
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(df[selected_features])
poly_feature_names = [
    f"poly_{name}" for name in poly.get_feature_names_out(selected_features)
]  # Add prefix to avoid column name conflicts

# Merge the original dataframe with the newly created polynomial features
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
df = pd.concat([df, df_poly], axis=1)

# Clean the dataset by replacing empty strings with NaN and logging missing values
df.replace("", np.nan, inplace=True)
logging.info(f"Missing values per column before dropna:\n{df.isnull().sum()}")

# Drop rows where all columns are NaN to clean the dataset further
df.dropna(how="all", inplace=True)
logging.info(f"Data shape after handling missing values: {df.shape}")

# Convert categorical variables to numeric codes to facilitate modeling
for feature in selected_features:
    if feature in df.columns:
        series = df[feature]
        logging.info(f"Type of {feature} before conversion: {type(series)}")
        logging.info(f"Data in {feature}: {series.head()}")
        if not isinstance(series.dtype, pd.CategoricalDtype):
            df[feature] = series.astype("category").cat.codes
            logging.info(f"Converted {feature} to categorical.")
        else:
            df[feature] = series.cat.codes
            logging.info(f"{feature} was already categorical.")
    else:
        logging.error(f"{feature} is not in DataFrame columns.")

# Define the features and target variable for the model
X = df.drop("v104", axis=1)  # Features
Y = df["v104"]  # Target variable

# Filter the dataset to include only classes with at least 10 samples
class_counts = Y.value_counts()
valid_classes = class_counts[class_counts >= 10].index
df_filtered = df[df["v104"].isin(valid_classes)]

# Redefine the features and target variable for the model using the filtered dataset
X_filtered = df_filtered.drop("v104", axis=1)  # Features
Y_filtered = df_filtered["v104"]  # Target variable

# Ensure that X_filtered contains only numeric columns before scaling
numeric_cols_filtered = X_filtered.select_dtypes(include=[np.number]).columns.tolist()
X_numeric_filtered = X_filtered[numeric_cols_filtered]

# Standardize features to normalize data distribution
scaler = StandardScaler()
X_scaled_filtered = scaler.fit_transform(X_numeric_filtered)

# Split the filtered dataset into training and testing sets
X_train_filtered, X_test_filtered, Y_train_filtered, Y_test_filtered = train_test_split(
    X_scaled_filtered, Y_filtered, test_size=0.2, stratify=Y_filtered, random_state=42
)
logging.info("Filtered data split into train and test sets.")

# Set up a pipeline with RandomForestClassifier and expanded hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200, 300, 500, 1000],
    "classifier__max_depth": [None, 10, 20, 30, 50],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 4, 6],
    "classifier__max_features": ["auto", "sqrt", "log2"],
}
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search_filtered = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="accuracy", verbose=3
)
grid_search_filtered.fit(X_train_filtered, Y_train_filtered)
best_model_filtered = grid_search_filtered.best_estimator_
logging.info(
    f"Expanded grid search best model parameters with filtered data: {grid_search_filtered.best_params_}"
)

# Evaluate the model on the filtered test set
Y_pred_filtered = best_model_filtered.predict(X_test_filtered)
accuracy_filtered = accuracy_score(Y_test_filtered, Y_pred_filtered)
logging.info(f"Model accuracy with filtered data: {accuracy_filtered}")
print(
    "Confusion Matrix with filtered data:\n",
    confusion_matrix(Y_test_filtered, Y_pred_filtered),
)
print(
    "Classification Report with filtered data:\n",
    classification_report(Y_test_filtered, Y_pred_filtered),
)
print(f"Accuracy of the model with filtered data: {accuracy_filtered:.2f}")

# Calculate and display the R^2 value for the model with filtered data
r2_value_filtered = r2_score(Y_test_filtered, Y_pred_filtered)
print(f"R^2 value with filtered data: {r2_value_filtered:.2f}")

# Save the trained model to a file for later use
model_filename_filtered = "output/forest_model_filtered.joblib"
joblib.dump(best_model_filtered, model_filename_filtered)
logging.info(f"Model saved to {model_filename_filtered}")
