# python scripts/main.py > output/output.log

import numpy as np
import pandas as pd
import pyreadstat
import joblib
import logging
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.calibration import CalibratedClassifierCV
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from dask_ml.preprocessing import Categorizer, DummyEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

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

    # Manually group v712 into bins and make it ordinal using CategoricalDtype
    bins = [-1, 1, 3, 5, 7, 9, 11]  # Define bin edges
    labels = [0, 1, 2, 3, 4, 5]  # Define labels for each bin
    df["v712_groups"] = pd.cut(df["v712"], bins=bins, labels=labels, ordered=True)
    df["v712_groups"] = df["v712_groups"].astype(pd.CategoricalDtype(ordered=True))

    # Adjust recode_v131 based on sector
    recode_map = {1.0: 1, 2.0: 2, 3.0: 3}
    df["recode_v131"] = df.apply(
        lambda row: 0 if row["sector"] == 2 else recode_map.get(row["v131"], 99), axis=1
    )

    # Remove 'Other' from sex
    df = df[df["sex"] != 3]

    # Group education into categories and make it ordinal
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
    df["educ_group"] = (
        df["educ"].map(education_map).astype(pd.CategoricalDtype(ordered=True))
    )

    # Group age into categories and make it ordinal
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
    df["age_group"] = (
        df["age_group"].map(age_map).astype(pd.CategoricalDtype(ordered=True))
    )

    selected_features = [
        "age_group",
        "v144",
        "v712_groups",
        "recode_v131",
        "sector",
        "sex",
        "educ_group",
        "v111",
    ]
    X = df[selected_features]
    y = df["v104"]

    # Define the categorical type for religiosity with explicit categories and make it ordinal
    cat_type_religiosity = pd.CategoricalDtype(
        categories=[1.0, 2.0, 3.0, 4.0, 5.0], ordered=True
    )
    df["v144"] = df["v144"].astype(cat_type_religiosity)  # religiosity

    # Define the categorical type for political spectrum with explicit categories and make it ordinal
    cat_type_political = pd.CategoricalDtype(
        categories=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ordered=True
    )
    df["v111"] = df["v111"].astype(cat_type_political)  # left-right political spectrum

    return X, y


def create_pipeline(min_samples):
    k_neighbors = min(min_samples - 1, 5)
    if min_samples <= 2:
        k_neighbors = 1

    # Feature reduction and interaction control
    categorical_transformer = Pipeline(
        [
            ("categorizer", Categorizer()),
            ("encoder", DummyEncoder()),
            ("scaler", StandardScaler()),  # Add a scaler here
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
        ]
    )

    # Enhanced preprocessor with feature selection
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                [
                    "age_group",
                    "v144",
                    "v712_groups",
                    "recode_v131",
                    "sex",
                    "educ_group",
                    "v111",
                ],
            ),
        ],
    )

    # Adjusted XGB parameters for further regularization
    xgb_params = {
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "max_depth": 6,
        "min_child_weight": 5,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "gamma": 1,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    xgb_classifier = XGBClassifier(**xgb_params)
    rf_classifier = RandomForestClassifier(n_estimators=200, max_features="sqrt")
    voting_clf = VotingClassifier(
        estimators=[
            ("xgb", xgb_classifier),
            ("rf", rf_classifier),
        ],
        voting="soft",
    )

    calibrated_clf = CalibratedClassifierCV(
        estimator=voting_clf, cv=5, method="sigmoid"
    )

    # Reintegrate BorderlineSMOTE
    smote = BorderlineSMOTE(
        sampling_strategy="auto", k_neighbors=k_neighbors, random_state=42
    )

    pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("resample", smote),
            ("classifier", calibrated_clf),
        ]
    )

    return pipeline


def merge_small_classes(y, threshold=10):
    class_counts = y.value_counts()
    small_classes = class_counts[class_counts < threshold].index

    while len(small_classes) > 0:
        smallest_class = small_classes[0]
        if len(small_classes) > 1:
            next_smallest_class = small_classes[1]
        else:
            next_smallest_class = class_counts[class_counts >= threshold].index[0]

        y = y.replace(smallest_class, next_smallest_class)
        class_counts = y.value_counts()
        small_classes = class_counts[class_counts < threshold].index

    return y


def main():
    filepath = "input/2022_SPSS.sav"
    X, y = load_and_prepare_data(filepath)

    # Split data by sector
    arab_data = X[X["sector"] == 2]
    jewish_data = X[X["sector"] == 1]
    y_arab = y[X["sector"] == 2]
    y_jewish = y[X["sector"] == 1]

    # Merge small classes in each sector
    y_arab_merged = merge_small_classes(y_arab, threshold=10)
    y_jewish_merged = merge_small_classes(y_jewish, threshold=10)

    # Encode labels
    encoder_arab = LabelEncoder()
    encoder_jewish = LabelEncoder()
    encoder_arab.fit(y_arab_merged.unique())
    encoder_jewish.fit(y_jewish_merged.unique())

    # Save the encoders
    joblib.dump(encoder_arab, "output/encoder_arab.joblib")
    joblib.dump(encoder_jewish, "output/encoder_jewish.joblib")

    y_arab_encoded = encoder_arab.transform(y_arab_merged)
    y_jewish_encoded = encoder_jewish.transform(y_jewish_merged)

    # Split data into training and testing sets for each sector
    X_arab_train, X_arab_test, y_arab_train, y_arab_test = train_test_split(
        arab_data.loc[y_arab_merged.index],
        y_arab_encoded,
        test_size=0.24,
        random_state=42,
        stratify=y_arab_encoded,
    )
    X_jewish_train, X_jewish_test, y_jewish_train, y_jewish_test = train_test_split(
        jewish_data.loc[y_jewish_merged.index],
        y_jewish_encoded,
        test_size=0.24,
        random_state=42,
        stratify=y_jewish_encoded,
    )

    # Determine the smallest class size in the training data using numpy
    unique, counts = np.unique(y_arab_train, return_counts=True)
    min_class_size_arab = counts.min()
    unique_jewish, counts_jewish = np.unique(y_jewish_train, return_counts=True)
    min_class_size_jewish = counts_jewish.min()

    # Create pipelines for each sector
    pipeline_arab = create_pipeline(min_samples=min_class_size_arab)
    pipeline_jewish = create_pipeline(min_samples=min_class_size_jewish)

    # Unified Optuna objective function
    def objective(trial, X_train, y_train, pipeline):
        param = {
            "classifier__estimator__xgb__max_depth": trial.suggest_int(
                "max_depth", 3, 15
            ),
            "classifier__estimator__xgb__min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 12
            ),
            "classifier__estimator__xgb__learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.2
            ),
            "classifier__estimator__xgb__n_estimators": trial.suggest_int(
                "n_estimators", 50, 1000
            ),
            "classifier__estimator__xgb__colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.3, 1.0
            ),
            "classifier__estimator__xgb__subsample": trial.suggest_float(
                "subsample", 0.4, 1.0
            ),
            "classifier__estimator__xgb__gamma": trial.suggest_float("gamma", 0, 5),
            "classifier__estimator__xgb__reg_alpha": trial.suggest_float(
                "reg_alpha", 0.01, 10
            ),
            "classifier__estimator__xgb__reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 10
            ),
            "classifier__estimator__xgb__max_delta_step": trial.suggest_int(
                "max_delta_step", 0, 10
            ),
        }
        pipeline.set_params(**param)
        try:
            skf = StratifiedKFold(n_splits=6)
            score = np.mean(
                cross_val_score(
                    pipeline, X_train, y_train, cv=skf, scoring="roc_auc_ovr"
                )
            )
        except ValueError as e:
            print(f"Error during scoring: {e}")
            return None
        return score

    # Optuna studies for each sector
    study_arab = optuna.create_study(direction="maximize")
    study_arab.optimize(
        lambda trial: objective(trial, X_arab_train, y_arab_train, pipeline_arab),
        n_trials=5,
    )

    study_jewish = optuna.create_study(direction="maximize")
    study_jewish.optimize(
        lambda trial: objective(trial, X_jewish_train, y_jewish_train, pipeline_jewish),
        n_trials=5,
    )

    # Correctly set parameters for the XGBClassifier within the VotingClassifier
    best_pipeline_arab = pipeline_arab.set_params(
        **{
            "classifier__estimator__xgb__" + key: value
            for key, value in study_arab.best_params.items()
        }
    )
    best_pipeline_jewish = pipeline_jewish.set_params(
        **{
            "classifier__estimator__xgb__" + key: value
            for key, value in study_jewish.best_params.items()
        }
    )

    # Train the best models
    best_pipeline_arab.fit(X_arab_train, y_arab_train)
    best_pipeline_jewish.fit(X_jewish_train, y_jewish_train)

    # Make predictions with the best models
    y_arab_pred = best_pipeline_arab.predict_proba(X_arab_test)
    y_jewish_pred = best_pipeline_jewish.predict_proba(X_jewish_test)

    # Evaluate models using both OVR and OVO
    arab_auc_ovr = roc_auc_score(y_arab_test, y_arab_pred, multi_class="ovr")
    arab_auc_ovo = roc_auc_score(y_arab_test, y_arab_pred, multi_class="ovo")
    jewish_auc_ovr = roc_auc_score(y_jewish_test, y_jewish_pred, multi_class="ovr")
    jewish_auc_ovo = roc_auc_score(y_jewish_test, y_jewish_pred, multi_class="ovo")

    # Log the AUC scores
    logging.info(f"Arab sector AUC OVR: {arab_auc_ovr}")
    logging.info(f"Arab sector AUC OVO: {arab_auc_ovo}")
    logging.info(f"Jewish sector AUC OVR: {jewish_auc_ovr}")
    logging.info(f"Jewish sector AUC OVO: {jewish_auc_ovo}")

    # Optionally, save the model
    joblib.dump(best_pipeline_arab, "output/best_model_arab.joblib")
    joblib.dump(best_pipeline_jewish, "output/best_model_jewish.joblib")


if __name__ == "__main__":
    main()
