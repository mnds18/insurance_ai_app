import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

def load_and_reindex(df, date_col="date", freq="D"):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "date"
    return df

def fill_missing(df, method="linear"):
    return df.interpolate(method=method).ffill().bfill()

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.where((series >= lower) & (series <= upper), np.nan)

def detrend(series, method="difference"):
    if method == "difference":
        return series.diff().dropna()
    elif method == "regression":
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        trend = model.predict(X).flatten()
        return pd.Series(y.flatten() - trend, index=series.index)
    else:
        raise ValueError("Unsupported detrending method")

def deseasonalize(series, period=7):
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    return res.trend + res.resid, res.seasonal, res

def smooth(series, method="rolling", window=7):
    if method == "rolling":
        return series.rolling(window=window, min_periods=1).mean()
    elif method == "ema":
        return series.ewm(span=window).mean()
    else:
        raise ValueError("Unsupported smoothing method")

def scale_series(series, scaler_type="minmax"):
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(scaled, index=series.index), scaler

def check_stationarity(series, alpha=0.05):
    result = adfuller(series.dropna())
    is_stationary = result[1] < alpha
    return is_stationary, result[1]

def preprocess_pipeline(df, value_col="value", plot=False, seasonal_period=7):
    df = load_and_reindex(df)
    df[value_col] = fill_missing(df[value_col])

    # Outlier handling
    df[value_col] = detect_outliers_iqr(df[value_col])
    df[value_col] = fill_missing(df[value_col])

    # Decomposition
    deseasonalized, seasonal, decomposition = deseasonalize(df[value_col], period=seasonal_period)

    # Detrending
    detrended = detrend(deseasonalized, method="regression")

    # Smoothing
    smoothed = smooth(detrended, method="rolling", window=7)

    # Stationarity check
    is_stationary, p_value = check_stationarity(smoothed)

    # Scaling
    scaled, scaler = scale_series(smoothed)

    if plot:
        plt.figure(figsize=(14, 10))
        plt.subplot(4, 1, 1); df[value_col].plot(title="Original")
        plt.subplot(4, 1, 2); seasonal.plot(title="Seasonal Component")
        plt.subplot(4, 1, 3); detrended.plot(title="Detrended")
        plt.subplot(4, 1, 4); scaled.plot(title="Smoothed + Scaled")
        plt.tight_layout()
        plt.show()

    return {
        "original": df[value_col],
        "deseasonalized": deseasonalized,
        "seasonal": seasonal,
        "detrended": detrended,
        "smoothed": smoothed,
        "scaled": scaled,
        "is_stationary": is_stationary,
        "p_value": p_value,
        "scaler": scaler,
        "residuals": decomposition.resid
    }

#  usage
if __name__ == "__main__":
    df = pd.read_csv("data/forecast_data_dirty.csv")
    result = preprocess_pipeline(df, value_col="total_payout", plot=True)

    cleaned_df = pd.DataFrame({
        "date": result["deseasonalized"].index,
        "total_payout": result["deseasonalized"].values,
        "claims_count": df.set_index("date").reindex(result["deseasonalized"].index)["claims_count"].interpolate(method="linear").ffill().bfill().values
    })

    cleaned_df.to_csv("data/forecast_data_clean.csv", index=False)
    print("✅ Cleaned multivariate forecast data saved to data/forecast_data_clean.csv")

