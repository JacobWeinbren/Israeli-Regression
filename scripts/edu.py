import pandas as pd
import pyreadstat
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    return df


def calculate_relationships(df, target):
    results = {}
    for column in df.columns:
        if column == target:
            continue
        # Encode all variables, assuming they are categorical
        encoder = LabelEncoder()
        valid_idx = df[column].notna() & df[target].notna()
        encoded_col = encoder.fit_transform(df.loc[valid_idx, column])
        encoded_target = encoder.fit_transform(df.loc[valid_idx, target])
        # Calculate Chi-squared test statistic and p-value for all variables
        chi2, p_value, _, _ = stats.chi2_contingency(
            pd.crosstab(encoded_col, encoded_target)
        )
        results[column] = (
            chi2,
            p_value,
        )  # Chi-squared statistic and p-value as measures of association
    # Sort results by the strength of relationship
    sorted_results = sorted(results.items(), key=lambda item: item[1][0], reverse=True)
    return sorted_results


def main():
    filepath = "input/2022_SPSS.sav"
    df = load_data(filepath)
    relationships = calculate_relationships(df, "v104")
    for variable, strength in relationships:
        print(f"{variable}: {strength}")


if __name__ == "__main__":
    main()
