import pandas as pd
import pyreadstat

# Load data
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")

# Analyze class distribution for each categorical variable
for column in ["v104"]:
    print(f"Distribution in {column}:")
    print(df[column].value_counts())
    print("\n")
