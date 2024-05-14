import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def rolling_windows(x, t, window_size, step_size=1):
    """Split time series data into rolling windows.
    
    The time series data (x) are expected to be an nD array, with n >= 1 and
    of the shape (# time stamps, ...); and the time stamps (t) in the form of
    a 1D-array are expected to be provided separately.
    
    The outputs are the windowed data (x_win) in the form of an (n+1)D array
    of the shape (# windows, window size, ...); and start times of the windows
    (t_win).
    """
    assert len(x) == len(t)
    n_windows = (len(x) - window_size) // step_size + 1
    x_win = np.full((n_windows, window_size, *(x.shape[1:])), np.nan)
    for i in range(window_size):
        x_win[:, i] = x[i::step_size][:n_windows]
    t_win = t[::step_size][:n_windows]
    return x_win, t_win
        

def apply_pipeline(x, freq, window_size):

    '''Apply a pipeline of preprocessing steps to transform raw time series 
    data into the required format for the reconstruction model (TadGAN).
    
    Parameters
    ----------
    x : pandas series indexed by datetimes
        Time series data.
    
    freq : str
        Resampling frequency, of the same format as the first argument of
        pandas.Series.resample().
    
    window_size : int
        Size of rolling windows of the preprocessed time series.
    
    Returns
    -------
    x : 1D-array
        Preprocessed time series.
                                                    
    t : 1D-array
        Time stamps of x.
    
    x_win : 2D-array
        Rolling windows of the preprocessed time series, of the shape
        (# windows, window size).
    
    t_win : 1D-array
        Starting time stamps of the rolling windows.
    '''

    x = x.resample(freq).sum()
    x, t = x.to_numpy(), x.index.to_numpy()
    x = x.reshape(-1, 1)
    x = SimpleImputer().fit_transform(x)
    x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
    x = x.reshape(-1)
    x_win, t_win = rolling_windows(x, t, window_size)
    
    return x, t, x_win, t_win

def new_apply_pipeline(x, freq, window_size):
    x_filled = x.ffill().bfill()  # Điền giá trị thiếu (nếu có)
    x = x_filled.to_numpy()
    t = x_filled.index.to_numpy()
    x = x.reshape(-1, 1)
    x_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)  # Scaling dữ liệu
    x_scaled = x_scaled.reshape(-1)
    x_win, t_win = rolling_windows(x_scaled, t, window_size)  # Tạo rolling windows
    
    return x_scaled, t, x_win, t_win

# def inverse_apply_pipeline(x, t, x_win, t_win, window_size):
#     # Reshape x_win to the original format
#     x_reconstructed = merge_rolling_windows(x_win)
    
#     # Reshape x to the original format
#     x = x.reshape(-1, 1)
#     x = MinMaxScaler(feature_range=(-1, 1)).inverse_transform(x)
#     x = x.flatten()
    
#     # Convert t to the original format
#     t = pd.to_datetime(t)
    
#     # Resample x to the original format
#     x_series = pd.Series(data=x, index=t)
#     x_resampled = x_series.resample('1D').sum()  # Use the same frequency used in apply_pipeline
    
#     return x_resampled.values, x_reconstructed, t, t_win