from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    LabelEncoder,
    OneHotEncoder,
)
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
import pyreadstat
import joblib

# Custom logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_prepare_data(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    logging.info(f"Data loaded with shape {df.shape}")

    # Filter out unwanted 'don't know' and 'other' categories
    df = df[df["v712"] != 98]  # Remove 'Don't know' from v712
    df = df[df["v131"] != 98]  # Remove 'Don't know' from v131
    unwanted_v104 = [30, 94, 96, 97, 98, 99]
    df = df[~df["v104"].isin(unwanted_v104)]  # Remove specified categories from v104

    logging.info(f"Data after filtering unwanted categories: {df.shape}")

    # Ensuring each class has at least a minimum number of samples
    min_samples_per_class = 10
    class_counts = df["v104"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df["v104"].isin(valid_classes)]
    logging.info(f"Data after filtering small classes: {df.shape}")

    recode_map = {1.0: 1, 2.0: 2, 3.0: 3, 5.0: 5, 10.0: 10}
    # Adjust recode_v131 based on sector
    df["recode_v131"] = df.apply(
        lambda row: 0 if row["sector"] == 2 else recode_map.get(row["v131"], 99), axis=1
    )

    selected_features = [
        "age_group",
        "v144",
        "v712",
        "recode_v131",
        "sector",
        "sex",
        "educ",
    ]
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

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["age_group", "v144", "v712"]),
            (
                "cat",
                categorical_transformer,
                ["recode_v131", "sector", "sex", "educ"],
            ),
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
        "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
        "classifier__learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
        "classifier__n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 2, 3, 4, 5, 6],
        "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "classifier__reg_alpha": [0, 0.1, 0.5, 1, 1.5, 2],
        "classifier__reg_lambda": [0.5, 1, 1.5, 2, 2.5, 3],
    }

    scorer = make_scorer(
        roc_auc_score, multi_class="ovo", response_method="predict_proba"
    )
    cv_strategy = StratifiedKFold(n_splits=10)

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=1000,
        scoring=scorer,
        cv=cv_strategy,
        verbose=3,
        error_score="raise",
        n_jobs=-1,
        random_state=42,
    )

    try:
        search.fit(X_resampled, y_resampled)  # Use resampled data for fitting
        best_model = search.best_estimator_

        # Save the pipeline
        joblib.dump(best_model, "output/pipeline.joblib")

        y_pred_proba = best_model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class="ovo")
        logging.info(f"Test AUC: {auc_score}")
    except Exception as e:
        logging.error(f"Error during model fitting: {e}")
        raise


if __name__ == "__main__":
    main()
