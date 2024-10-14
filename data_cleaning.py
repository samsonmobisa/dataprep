import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the input DataFrame by handling missing values,
    normalizing formats, and flagging outliers.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Fill missing values with forward fill
    df.fillna(method='ffill', inplace=True)

    # Normalize data format (example column name 'column_name')
    if 'column_name' in df.columns:
        df['column_name'] = df['column_name'].str.lower()

    # Flag outliers using Z-score method
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std_dev = df[col].std()
        threshold = 3 * std_dev
        df[f'{col}_outlier'] = np.abs(df[col] - mean) > threshold

    return df
