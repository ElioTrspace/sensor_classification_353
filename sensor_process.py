import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
import os
from datetime import datetime, timedelta
import re

def combine_sensor_files(sensor_files):
    dfs = [pd.read_csv(f) for f in sensor_files]
    return pd.concat(dfs, ignore_index=True)

def sensor_time_to_datetime(df, file_path):
    """Convert sensor data to datetime using file name timestamp."""
    fname = os.path.basename(file_path)
    # Hardcode to match YYYY-MM-DDHH.MM.SS pattern
    m = re.search(r'(\d{4}-\d{2}-\d{2})(\d{2}\.\d{2}\.\d{2})', fname)
    if not m:
        raise ValueError(f"Filename {fname} does not match expected date/time pattern")
    
    date_part, time_part = m.groups()
    dt = datetime.strptime(date_part + ' ' + time_part, '%Y-%m-%d %H.%M.%S')
    
    if 'time_sec' in df.columns:
        df['datetime'] = df['time_sec'].apply(lambda x: dt + timedelta(seconds=x))
    else:
        df['datetime'] = [dt + timedelta(seconds=i) for i in range(len(df))]
    
    return df

def add_day_night(df, sunset_hour=20, sunset_minute=45):
    """Add day/night dummies from absolute datetimes."""
    if "datetime" not in df.columns:
        raise KeyError("Expected 'datetime' on sensor df before add_day_night().")

    def _dn(x):
        return "night" if (x.hour > sunset_hour or (x.hour == sunset_hour and x.minute >= sunset_minute)) else "day"

    df["day_or_night"] = df["datetime"].apply(_dn)
    return pd.get_dummies(df, columns=["day_or_night"])

def butter_lowpass_filter(df: pd.DataFrame, cols, cutoff=5.0, fs=50.0, order=3):
    """
    Low-pass Butterworth across the requested columns (in-place).
    """
    nyq = 0.5 * fs
    wn = min(0.999999, max(1e-6, cutoff / nyq))
    b, a = butter(order, wn, btype="low", analog=False)

    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").interpolate(limit_direction="both").to_numpy()
            df[c] = filtfilt(b, a, x)
    return df


def drop_unwanted_sensor_cols(df: pd.DataFrame):
    to_drop = [c for c in ["Azimuth", "Pitch", "Roll"] if c in df.columns]
    if to_drop:
        df = df.drop(columns = to_drop)
    # Also nuke unnamed columns often created by CSV writes
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns = unnamed)
    return df