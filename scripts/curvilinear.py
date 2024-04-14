import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import pyreadstat
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load data
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")
logging.info(f"Data loaded with shape {df.shape}")

# Select and prepare the scale variable
df["v712"] = df["v712"].replace(98, np.nan)  # Corrected inplace operation
df.dropna(subset=["v712"], inplace=True)
df["v712"] = df["v712"].astype(float)

# Generate polynomial features (quadratic)
poly = PolynomialFeatures(degree=2, include_bias=True)
v712_poly = poly.fit_transform(df[["v712"]])
poly_features = poly.get_feature_names_out(["v712"])

# Debugging: Print the number of polynomial features and names
print("Number of polynomial features:", len(poly_features))
print("Polynomial feature names:", poly_features)

# Create a DataFrame for polynomial features
v712_poly_df = pd.DataFrame(v712_poly, columns=poly_features)
df = pd.concat([df, v712_poly_df], axis=1)

# Remove any duplicate columns that might have been created inadvertently
df = df.loc[:, ~df.columns.duplicated()]

# Handle NaN values in features
imputer = SimpleImputer(strategy="mean")
df[poly_features] = imputer.fit_transform(df[poly_features])

# Ensure no NaN values in target variable
df.dropna(subset=["v104"], inplace=True)

# Fit a linear model without intercept
X = df[poly_features]
Y = df["v104"]

# Debugging: Print X columns before fitting the model
print("Columns used in model fitting:", X.columns)

model = LinearRegression(fit_intercept=False)
model.fit(X, Y)

# Debugging: Print the shape and actual coefficients of the model
print("Shape of coefficients:", model.coef_.shape)
print("Coefficients:", model.coef_)

# Evaluate the model
Y_pred = model.predict(X)
r2 = r2_score(Y, Y_pred)
logging.info(f"Model R^2 value: {r2:.2f}")

# Output the coefficients for the polynomial terms
coefficients = pd.DataFrame(
    model.coef_.reshape(-1, 1), index=poly_features, columns=["Coefficient"]
)
print("Coefficients for polynomial terms:\n", coefficients)

# Save the model coefficients to a file
coefficients.to_csv("output/v712_polynomial_coefficients.csv")
logging.info("Model coefficients saved to output/v712_polynomial_coefficients.csv")
