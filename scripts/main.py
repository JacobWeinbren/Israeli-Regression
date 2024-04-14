import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE
import pyreadstat
import logging
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load data from .sav file
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")
logging.info(f"Data loaded with shape {df.shape}")

# Define the specific features to be used
selected_features = ["age_group", "sex", "v144", "v712", "v131", "v143_code"]

# Update DataFrame to keep only the selected features and the target variable
df = df[selected_features + ["v104"]]  # Assuming 'v104' is the target variable
logging.info("Limited dataset to selected features")

# Convert all empty strings to NaN
df.replace("", np.nan, inplace=True)

# Drop rows with missing values in key columns
df = df.dropna()
logging.info("Dropped rows with missing values in key columns.")

# Convert categorical variables to codes
for feature in selected_features:
    df[feature] = df[feature].astype("category").cat.codes
logging.info("Converted categorical variables to codes.")

# Define features and target
X = df.drop("v104", axis=1)
Y = df["v104"]

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors="coerce")
logging.info("Ensured all columns are numeric.")

# Handle NaN values using an imputer
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
logging.info("Applied mean imputation.")

# Create a DataFrame from the imputed data
X_imputed_df = pd.DataFrame(X_imputed, columns=selected_features)
logging.info("DataFrame created from imputed data.")

# Continue with scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
logging.info("Data scaling applied.")

# Handling imbalanced data
smote = SMOTE(k_neighbors=1)
X_smote, Y_smote = smote.fit_resample(X_scaled, Y)
logging.info("Applied SMOTE with adjusted n_neighbors to handle class imbalance.")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_smote, Y_smote, test_size=0.2, stratify=Y_smote, random_state=42
)
logging.info("Data split into train and test sets.")

# Hyperparameter tuning
param_grid = {
    "n_estimators": [300, 600, 900],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", verbose=2)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
logging.info(f"Best model parameters: {grid_search.best_params_}")

# Predict and evaluate
Y_pred_proba = best_model.predict_proba(X_test)
print("Probabilities of voting for each party:\n", Y_pred_proba)

# You can still calculate accuracy based on the highest probability prediction if needed
Y_pred = np.argmax(Y_pred_proba, axis=1)
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

# Save the model to a file
model_filename = "output/forest_model.joblib"
joblib.dump(best_model, model_filename)
logging.info(f"Model saved to {model_filename}")
