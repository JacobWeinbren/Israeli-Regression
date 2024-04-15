from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import roc_auc_score, make_scorer
import logging
import numpy as np
import pandas as pd
import pyreadstat

# Custom logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_data(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    logging.info(f"Data loaded with shape {df.shape}")

    # Ensuring each class has at least a minimum number of samples
    min_samples_per_class = 10
    class_counts = df["v104"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df["v104"].isin(valid_classes)]
    logging.info(f"Data after filtering small classes: {df.shape}")

    recode_map = {1.0: 1, 2.0: 2, 3.0: 3, 5.0: 5, 10.0: 10}
    df["recode_v131"] = df["v131"].apply(lambda x: recode_map.get(x, 99))

    selected_features = ["age_group", "v144", "v712", "recode_v131", "sector"]
    X = df[selected_features]
    y = df["v104"]

    return X, y


def create_pipeline():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=2)),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["age_group", "v144", "v712", "recode_v131"]),
        ]
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(eval_metric="mlogloss", use_label_encoder=False),
            ),
        ]
    )

    return pipeline


def main():
    filepath = "input/2022_SPSS.sav"
    X, y = load_and_prepare_data(filepath)

    # Define all possible classes explicitly
    all_classes = np.unique(y)
    encoder = LabelEncoder()
    encoder.fit(all_classes)
    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Manually handle resampling
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Ensure no class mismatch
    if not np.array_equal(np.unique(y_resampled), np.unique(y_train)):
        logging.error("Class mismatch detected after resampling")
        raise ValueError("Resampling resulted in inconsistent class labels.")

    pipeline = create_pipeline()
    pipeline.named_steps["classifier"].fit(X_resampled, y_resampled)

    param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "classifier__n_estimators": [50, 100, 150, 200, 250, 300],
        "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    scorer = make_scorer(
        roc_auc_score, multi_class="ovo", response_method="predict_proba"
    )
    cv_strategy = StratifiedKFold(n_splits=5)

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=50,
        scoring=scorer,
        cv=cv_strategy,
        verbose=3,
        error_score="raise",
        n_jobs=-1,
    )

    try:
        search.fit(X_resampled, y_resampled)  # Use resampled data for fitting
        best_model = search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class="ovo")
        logging.info(f"Test AUC: {auc_score}")
    except Exception as e:
        logging.error(f"Error during model fitting: {e}")
        raise


if __name__ == "__main__":
    main()
