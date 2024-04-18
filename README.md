# Codebase Overview

This codebase is structured to handle data processing, model training, and prediction for machine learning on the [Israeli Election Study 2022](https://socsci4.tau.ac.il/mu2/ines/). It includes scripts for data preparation, model training, evaluation, and a web application for making predictions.

## Key Components

### Data Preparation and Analysis

-   **Data Loading and Filtering**: Data is loaded from SPSS `.sav` files using `pyreadstat` and filtered based on specific criteria.

```bash
python scripts/main.py
```

-   **Class Distribution Analysis**: Analyzes the class distribution for categorical variables.

```bash
python scripts/distribution.py
```

### Model Training and Evaluation

-   **Pipeline Creation**: A pipeline is created for preprocessing and model training using XGBoost and RandomForest within a VotingClassifier.

-   **Hyperparameter Optimization**: Optuna is used for hyperparameter optimisation of the XGBoost model.

-   **Model Evaluation**: Models are evaluated using ROC AUC scores for both One-vs-Rest (OVR) and One-vs-One (OVO) strategies.

-   **Prediction Script**: A standalone script for making predictions from the command line.

```bash
python scripts/prediction.py
```

### Frontend

-   **Astro-based Frontend**: The frontend is built using Astro, showcasing components for form inputs and displaying prediction results.

```bash
cd site
npm run dev
```

## Dependencies

The project relies on several key Python libraries, including `Flask`, `joblib`, `imblearn`, `xgboost`, `pandas`, `Flask-Cors`, `dask-ml`, and `Gunicorn`, as specified in `requirements.txt`.

## License

The project is licensed under the MIT License, allowing for wide use and modification.
