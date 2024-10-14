import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_outliers(df):
    """
    Detects outliers in the DataFrame using Isolation Forest.

    Parameters:
    df (pd.DataFrame): The input DataFrame with numerical values.

    Returns:
    pd.DataFrame: DataFrame with outlier predictions.
    """
    model = IsolationForest(contamination=0.1)  # Adjust contamination as needed
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df['outlier'] = model.fit_predict(df[numerical_cols])
    return df
