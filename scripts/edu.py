import statsmodels.api as sm
import logging
import pyreadstat

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")
logging.info(f"Data loaded with shape {df.shape}")

import numpy as np

# Assuming 'educ' is education and 'v143_code' is socio-economic ranking
# and 'v104' is your target variable.
# Check if 'v104' is binary
if not set(df["v104"].unique()).issubset({0, 1}):
    # Convert 'v104' to binary if not already (example conversion)
    threshold = 10  # Define your own threshold
    df["v104"] = (df["v104"] > threshold).astype(int)

print(df.columns.tolist())

# Prepare the data
selected_features = [
    "age_group",
    "sex",
    "v144",
    "v712",
    "v131",
    "v143_code",
    "educ",
]
X = df[selected_features]
Y = df["v104"]  # Target variable

# Check for NaNs or infinite values and handle them
X = X.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
X = X.dropna()  # Drop rows with NaNs
Y = Y.loc[X.index]  # Ensure Y is aligned with X after dropping rows

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(Y, X)
result = model.fit()

# Print the summary of the regression results
print(result.summary())
