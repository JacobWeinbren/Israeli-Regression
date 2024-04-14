import pandas as pd
import pyreadstat

# Load data
df, meta = pyreadstat.read_sav("input/2022_SPSS.sav")

# Analyze class distribution for each categorical variable
for column in ["age_group", "v144", "v712", "v131", "v143_code", "sector"]:
    print(f"Distribution in {column}:")
    print(df[column].value_counts())
    print("\n")
