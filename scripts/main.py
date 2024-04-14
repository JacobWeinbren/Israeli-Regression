import pandas as pd
import numpy as np
import pyreadstat
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_data(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    logging.info(f"Data loaded with shape {df.shape}")

    # Apply filters and recode data
    df = df[(df["v712"] != 98) & (~df["v104"].isin([30, 94, 96, 97, 98, 99]))]
    valid_v131 = df["v131"].value_counts()
    valid_v131 = valid_v131[valid_v131 >= 10].index.difference([98.0])
    df = df[df["v131"].isin(valid_v131)]
    df["recode_v131"] = df.apply(
        lambda row: row["v131"] if row["sector"] != "Arab" else 0, axis=1
    )
    df["recode_v131"] = (
        df["recode_v131"]
        .replace({1.0: 1, 2.0: 2, 3.0: 3, 5.0: 5, 10.0: 10, 0.0: 0})
        .fillna(99)
    )

    # Filter by class size
    target_column = "v104"
    valid_classes = df[target_column].value_counts()
    valid_classes = valid_classes[valid_classes >= 10].index
    df = df[df[target_column].isin(valid_classes)]
    logging.info(f"Data after filtering: {df.shape}")

    return df, target_column


def main():
    filepath = "input/2022_SPSS.sav"
    selected_features = ["age_group", "v144", "v712", "recode_v131", "sector"]

    df, target_column = load_and_prepare_data(filepath)
    X, y = df[selected_features], df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Calculate k_neighbors based on the smallest class count in the training set
    min_class_count = y_train.value_counts().min()
    k_neighbors = max(
        min_class_count - 1, 1
    )  # Ensure k_neighbors is at least 1 and less than min_class_count
    logging.info(f"Adjusted SMOTE k_neighbors for training set: {k_neighbors}")

    pipeline = create_pipeline(selected_features, target_column, k_neighbors)

    # Grid search for hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 500, 1000, 1500, 2000],
        "classifier__max_depth": [None, 5, 10, 20, 30, 40, 50, 60],
        "classifier__min_samples_split": [2, 5, 10, 15, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 6, 8],
        "classifier__max_features": ["auto", "sqrt", "log2", None],
        "smote__k_neighbors": [1, 3, 5, 7, 10, 15, 20],
        "preprocessor__num__imputer__strategy": ["mean", "median", "most_frequent"],
        "feature_selection__step": [1, 2, 3],
        "feature_selection__min_features_to_select": [1, 2, 3, 5],
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="balanced_accuracy", verbose=3
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    joblib.dump(best_pipeline, "output/pipeline.joblib")
    logging.info("Best pipeline trained and saved.")

    # Predict on the test set using the best pipeline
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test accuracy with optimized pipeline: {accuracy}")


def create_pipeline(selected_features, target_column, k_neighbors):
    # Preprocessing for numerical features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine all preprocessing
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, selected_features)]
    )

    # Feature selection
    rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=5, scoring="accuracy")

    # Create a complete pipeline
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("feature_selection", rfecv),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42, n_jobs=-1, class_weight="balanced"
                ),
            ),
        ]
    )

    return pipeline


if __name__ == "__main__":
    main()
