import joblib
import pandas as pd
import pyreadstat

# Load the dataset
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")

# Filter for the Arab sector
arab_data = df[df["sector"] == 2]

# Load the Arab model and the encoder used during training
arab_model = joblib.load("output/best_model_arab.joblib")
encoder_arab = joblib.load("output/encoder_arab.joblib")

# Use the loaded encoder to transform data or inverse transform model classes
model_classes = arab_model.classes_
original_classes = encoder_arab.inverse_transform(model_classes)
print("Classes predicted by the Arab model (original labels):")
print(original_classes)

# Determine which classes have been removed
all_classes = set(arab_data["v104"].unique())
kept_classes = set(original_classes)
removed_classes = all_classes - kept_classes

print("Classes kept in the model (original labels):", kept_classes)
print("Classes removed from the model (original labels):", removed_classes)
