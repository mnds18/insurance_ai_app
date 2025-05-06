import pandas as pd
import numpy as np

def create_lag_features(df, target_col="total_payout", lags=[1, 7, 30]):
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        df[f"{target_col}_diff_{lag}"] = df[target_col] - df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col="total_payout", windows=[7, 30]):
    for window in windows:
        df[f"{target_col}_rollmean_{window}"] = df[target_col].rolling(window).mean()
        df[f"{target_col}_rollstd_{window}"] = df[target_col].rolling(window).std()
    return df

def add_date_parts(df, date_col="date"):
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df[date_col].dt.month
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    return df

def train_test_split_time_series(df, date_col="date", test_size=0.2):
    df = df.sort_values(by=date_col)
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(rmse, 2), round(mape, 2)
