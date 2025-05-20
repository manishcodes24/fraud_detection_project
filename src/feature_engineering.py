import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    """
    Scales 'Amount' and 'Time' columns using StandardScaler.
    Returns a new DataFrame with scaled features.
    """
    df_scaled = df.copy()
    
    scaler = StandardScaler()
    df_scaled[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])

    # Drop original 'Amount' and 'Time'
    df_scaled.drop(['Amount', 'Time'], axis=1, inplace=True)

    return df_scaled
