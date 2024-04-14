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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pyreadstat
import logging
import joblib

# Configure logging to monitor progress and debug issues
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    logging.info(f"Data loaded with shape {df.shape}")
    return df


def preprocess_data(df, selected_features):
    # Remove non-numeric columns first
    df = df.select_dtypes(include=[np.number, "category"])

    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Create polynomial features for selected features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    selected_data = df[selected_features]
    poly_features = poly.fit_transform(selected_data)
    poly_feature_names = [
        f"poly_{name}" for name in poly.get_feature_names_out(selected_features)
    ]
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = pd.concat([df, df_poly], axis=1)

    # Convert categorical variables to numeric codes
    for feature in selected_features:
        if feature in df.columns:
            series = df[feature]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                df[feature] = series.astype("category").cat.codes
            else:
                df[feature] = series.cat.codes

    logging.info(f"Data shape after preprocessing: {df.shape}")
    return df


def setup_model(X, Y):
    # Determine the smallest class size in Y
    min_class_size = Y.value_counts().min()
    k_neighbors = max(1, min_class_size - 1)  # Ensure at least one neighbor

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    try:
        X_resampled, Y_resampled = smote.fit_resample(X, Y)
    except ValueError as e:
        logging.error(f"SMOTE error: {str(e)}")
        return None

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_resampled, test_size=0.2, random_state=42
    )

    # Define the model pipeline with imbalance handling
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42, n_jobs=-1, class_weight="balanced"
                ),
            ),
        ]
    )

    # Hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": [100, 300, 500, 700, 1000, 1300, 1500],
        "classifier__max_depth": [None, 10, 20, 30, 40],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["auto", "sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        verbose=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")

    return best_model, X_test, Y_test


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    logging.info(f"Model accuracy: {accuracy}")
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print(f"Accuracy of the model: {accuracy:.2f}")

    r2_value = r2_score(Y_test, Y_pred)
    print(f"R^2 value: {r2_value:.2f}")


def save_model(model, filename):
    joblib.dump(model, filename)
    logging.info(f"Model saved to {filename}")


def main():
    df = load_data("input/2022_SPSS.sav")
    selected_features = ["age_group", "v144", "v712", "v131", "v143_code"]
    df = preprocess_data(df, selected_features)
    X = df.drop("v104", axis=1)
    Y = df["v104"]

    # Explicitly check for NaNs in Y
    if Y.isnull().any():
        logging.error("NaN values detected in target variable Y.")
        return

    best_model, X_test, Y_test = setup_model(X, Y)
    if best_model is not None:
        evaluate_model(best_model, X_test, Y_test)
        save_model(best_model, "output/forest_model_filtered.joblib")


if __name__ == "__main__":
    main()
