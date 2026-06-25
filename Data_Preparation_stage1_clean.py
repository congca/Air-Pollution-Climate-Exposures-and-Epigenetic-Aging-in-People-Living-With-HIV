import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def expand_features(csv_input, csv_output, degree=2):
    # Load file
    df = pd.read_csv(csv_input)

    # Separate features and missing values
    missing_indicator = df.isnull()
    features = df.fillna(0)

    # Polynomial feature expansion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    expanded_features = poly.fit_transform(features)

    # DataFrame with expanded features
    expanded_df = pd.DataFrame(expanded_features, columns=poly.get_feature_names_out(df.columns))

    # Reapply the missing value indicators
    expanded_df[missing_indicator] = None

    # Save to new file
    expanded_df.to_csv(csv_output, index=False)

expand_features('Cleaned.csv', 'Expansion.csv', degree=2)
from fancyimpute import IterativeSVD
from sklearn.impute import KNNImputer
input_csv_path = 'Expansion_Cleaned.csv'
df = pd.read_csv(input_csv_path)

# SVD imputation
imputer = IterativeSVD()
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Save to new CSV file
output_csv_path = 'Impute.csv'
df_imputed.to_csv(output_csv_path, index=False)

print(f"\nImputed data saved to {output_csv_path}")
from sklearn.preprocessing import MinMaxScaler

input_csv_path = 'Impute.csv'
output_csv_path = 'Normalization.csv'

df = pd.read_csv(input_csv_path)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize DataFrame
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save to new CSV file
df_normalized.to_csv(output_csv_path, index=False)

print(f"\nNormalized data saved to {output_csv_path}")