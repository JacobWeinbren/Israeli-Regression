# python scripts/main.py > output/output.log

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
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
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

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
    unwanted_v104 = [16, 17, 18, 19, 20, 21, 30, 94, 96, 97, 98, 99]
    df = df[~df["v104"].isin(unwanted_v104)]  # Remove specified categories from v104

    logging.info(f"Data after filtering unwanted categories: {df.shape}")

    # Manually group v712 into bins
    bins = [-1, 1, 3, 5, 7, 9, 11]  # Define bin edges
    labels = [0, 1, 2, 3, 4, 5]  # Define labels for each bin
    df["v712_groups"] = pd.cut(df["v712"], bins=bins, labels=labels)

    # Adjust recode_v131 based on sector
    recode_map = {1.0: 1, 2.0: 2, 3.0: 3}
    df["recode_v131"] = df.apply(
        lambda row: 0 if row["sector"] == 2 else recode_map.get(row["v131"], 99), axis=1
    )

    # Remove 'Other' from sex
    df = df[df["sex"] != 3]

    # Group education into categories
    education_map = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 2,
        8: 2,
        9: 3,
    }
    df["educ_group"] = df["educ"].map(education_map)

    # Group age into categories
    age_map = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
    }
    df["age_group"] = df["age_group"].map(age_map)

    selected_features = [
        "age_group",
        "v144",
        "v712_groups",
        "recode_v131",
        "sector",
        "sex",
        "educ_group",
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
            ("num", numeric_transformer, ["age_group", "v144", "v712_groups"]),
            ("cat", categorical_transformer, ["recode_v131", "sex", "educ_group"]),
        ]
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    max_depth=10,
                    n_estimators=300,
                    learning_rate=0.05,
                ),
            ),
        ]
    )

    return pipeline


def main():
    filepath = "input/2022_SPSS.sav"
    X, y = load_and_prepare_data(filepath)

    # Split data by sector
    arab_data = X[X["sector"] == 2]
    jewish_data = X[X["sector"] == 1]
    y_arab = y[X["sector"] == 2]
    y_jewish = y[X["sector"] == 1]

    # Filter out classes with fewer than two samples in each sector
    vc_arab = y_arab.value_counts()
    y_arab = y_arab[y_arab.isin(vc_arab[vc_arab >= 2].index)]
    arab_data = arab_data.loc[y_arab.index]

    vc_jewish = y_jewish.value_counts()
    y_jewish = y_jewish[y_jewish.isin(vc_jewish[vc_jewish >= 2].index)]
    jewish_data = jewish_data.loc[y_jewish.index]

    # Encode labels
    encoder_arab = LabelEncoder()
    encoder_jewish = LabelEncoder()
    encoder_arab.fit(y_arab.unique())
    encoder_jewish.fit(y_jewish.unique())

    # Save the encoders
    joblib.dump(encoder_arab, "output/encoder_arab.joblib")
    joblib.dump(encoder_jewish, "output/encoder_jewish.joblib")

    y_arab_encoded = encoder_arab.transform(y_arab)
    y_jewish_encoded = encoder_jewish.transform(y_jewish)

    # Split data into training and testing sets for each sector
    X_arab_train, X_arab_test, y_arab_train, y_arab_test = train_test_split(
        arab_data,
        y_arab_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_arab_encoded,
    )
    X_jewish_train, X_jewish_test, y_jewish_train, y_jewish_test = train_test_split(
        jewish_data,
        y_jewish_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_jewish_encoded,
    )

    # Determine the smallest class size in the training data using numpy
    unique, counts = np.unique(y_arab_train, return_counts=True)
    min_class_size_arab = counts.min()
    unique_jewish, counts_jewish = np.unique(y_jewish_train, return_counts=True)
    min_class_size_jewish = counts_jewish.min()

    # Set n_neighbors to one less than the number of samples in the smallest class
    min_k_neighbors = 5  # Set a minimum threshold for k_neighbors
    smote_arab = BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=min(max(1, min_class_size_arab - 1), min_k_neighbors),
        kind="borderline-1",  # or "borderline-2" based on specific needs
    )
    smote_jewish = BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=min(max(1, min_class_size_jewish - 1), min_k_neighbors),
        kind="borderline-1",  # or "borderline-2"
    )

    # Apply SMOTE
    X_arab_resampled, y_arab_resampled = smote_arab.fit_resample(
        X_arab_train, y_arab_train
    )
    X_jewish_resampled, y_jewish_resampled = smote_jewish.fit_resample(
        X_jewish_train, y_jewish_train
    )

    # Create pipelines for each sector
    pipeline_arab = create_pipeline()
    pipeline_jewish = create_pipeline()

    # Define parameter grid
    param_grid = {
        "classifier__max_depth": [3, 5, 7, 9, 12, 15],
        "classifier__min_child_weight": [1, 2, 3, 4, 5, 6, 8, 10],
        "classifier__learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25],
        "classifier__n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "classifier__colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "classifier__reg_alpha": [0, 0.1, 0.5, 1, 1.5, 2],
        "classifier__reg_lambda": [0.5, 1, 1.5, 2, 2.5, 3],
    }

    # Setup RandomizedSearchCV for each sector
    scorer = make_scorer(
        roc_auc_score, multi_class="ovo", response_method="predict_proba"
    )

    # Determine the smallest class size in the training data using numpy
    unique_arab, counts_arab = np.unique(y_arab_train, return_counts=True)
    min_class_size_arab = counts_arab.min()
    unique_jewish, counts_jewish = np.unique(y_jewish_train, return_counts=True)
    min_class_size_jewish = counts_jewish.min()

    cv_strategy_arab = StratifiedKFold(n_splits=min_class_size_arab)
    cv_strategy_jewish = StratifiedKFold(n_splits=min_class_size_jewish)

    search_arab = RandomizedSearchCV(
        pipeline_arab,
        param_grid,
        n_iter=10000,
        scoring=scorer,
        cv=cv_strategy_arab,
        verbose=3,
        error_score="raise",
        n_jobs=-1,
        random_state=42,
    )
    search_jewish = RandomizedSearchCV(
        pipeline_jewish,
        param_grid,
        n_iter=10000,
        scoring=scorer,
        cv=cv_strategy_jewish,
        verbose=3,
        error_score="raise",
        n_jobs=-1,
        random_state=42,
    )

    # Train each model using RandomizedSearchCV
    search_arab.fit(X_arab_resampled, y_arab_resampled)
    search_jewish.fit(X_jewish_resampled, y_jewish_resampled)

    # Best models after hyperparameter tuning
    best_model_arab = search_arab.best_estimator_
    best_model_jewish = search_jewish.best_estimator_

    # Make predictions with the best models
    y_arab_pred = best_model_arab.predict_proba(X_arab_test)
    y_jewish_pred = best_model_jewish.predict_proba(X_jewish_test)

    # Evaluate models
    arab_auc = roc_auc_score(y_arab_test, y_arab_pred, multi_class="ovo")
    jewish_auc = roc_auc_score(y_jewish_test, y_jewish_pred, multi_class="ovo")

    logging.info(f"Arab sector AUC: {arab_auc}")
    logging.info(f"Jewish sector AUC: {jewish_auc}")

    # Save the best models
    joblib.dump(best_model_arab, "output/best_model_arab.joblib")
    joblib.dump(best_model_jewish, "output/best_model_jewish.joblib")


if __name__ == "__main__":
    main()
