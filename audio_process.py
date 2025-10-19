import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt
# import ffmpeg
from datetime import datetime, timedelta

# does not work unfortunately
# def convert_m4a_to_wav(m4a_path):
#     wav_path = m4a_path.rsplit('.', 1)[0] + '.wav'
#     if not os.path.exists(wav_path):
#         try:
#             ffmpeg.input(m4a_path).output(wav_path, loglevel='quiet').run(overwrite_output=True)
#         except ffmpeg.Error as e:
#             print(f"Error converting {m4a_path} to wav: {e}")
#             raise
#     return wav_path

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_audio_file(file_path, lowcut=20, highcut=20000):
    """
    Load a WAV, bandpass filter, compute per-second RMS.
    Returns DataFrame with columns: ['time_sec','rms','file'] (no datetime yet).
    """
    y, sr = librosa.load(file_path, sr = None, mono = True)
    y = butter_bandpass_filter(y, lowcut, highcut, sr)

    n_sec = int(len(y) // sr)
    if n_sec == 0:
        return pd.DataFrame(columns = ["time_sec", "rms", "file"])

    y = y[: n_sec * sr]
    y_2d = y.reshape(n_sec, sr)
    rms = np.sqrt((y_2d ** 2).mean(axis=1))

    df = pd.DataFrame({
        "time_sec": np.arange(n_sec, dtype=int),
        "rms": rms,
        "file": os.path.basename(file_path),
    })
    return df

def combine_audio_files(audio_files):
    """Stack per-file audio features into one DataFrame (no datetime yet)."""
    dfs = []
    for f in audio_files:
        df = process_audio_file(f)
        df["source_audio_path"] = f
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["time_sec", "rms", "file", "source_audio_path"])
    return pd.concat(dfs, ignore_index=True)


def audio_time_to_datetime(audio_df, filename):
    """
    Add absolute datetimes from audio filename pattern: MyRec_MMDD_HHMM.wav
    The timestamp is the END time; backfill using time_sec and total duration.
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Audio filename {base} not like 'MyRec_MMDD_HHMM.wav'")

    date_part = parts[1]           # '0805'
    time_part = parts[2].split(".")[0]  # '1628'

    month = int(date_part[:2])
    day = int(date_part[2:])
    hour = int(time_part[:2])
    minute = int(time_part[2:])

    end_dt = datetime(2025, month, day, hour, minute)

    file_mask = (audio_df["file"] == base)
    if not file_mask.any():
        return audio_df

    duration_sec = int(audio_df.loc[file_mask, "time_sec"].max())
    offset = audio_df.loc[file_mask, "time_sec"].astype(int)
    audio_df.loc[file_mask, "datetime"] = offset.apply(
        lambda s: end_dt - timedelta(seconds=(duration_sec - s))
    )
    return audio_df