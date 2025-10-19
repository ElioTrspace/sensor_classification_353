import os
import pandas as pd

from audio_process import combine_audio_files, audio_time_to_datetime
from sensor_process import (
    sensor_time_to_datetime,
    butter_lowpass_filter,
    add_day_night,
    drop_unwanted_sensor_cols,
)
from labels import read_label_files, merge_labels
from feature_merge import merge_audio_sensor
from model_train import evaluate_dual_sensors


PROJECT_DIR = "project_data"

audio_files = [os.path.join(PROJECT_DIR, f) for f in os.listdir(PROJECT_DIR) if f.lower().endswith(".wav")]

audio_df_list = []
for f in audio_files:
    df = combine_audio_files([f])
    df = audio_time_to_datetime(df, f)
    audio_df_list.append(df)

if audio_df_list:
    audio_df = pd.concat(audio_df_list, ignore_index=True).sort_values("datetime").reset_index(drop=True)
else:
    audio_df = pd.DataFrame(columns=["time_sec", "rms", "file", "source_audio_path", "datetime"])

audio_df["datetime"] = pd.to_datetime(audio_df["datetime"], errors="coerce")

sensor_files = [os.path.join(PROJECT_DIR, f) for f in os.listdir(PROJECT_DIR) if f.lower().endswith(".csv")]
sensor_df_list = []
for f in sensor_files:
    df = pd.read_csv(f)
    df = sensor_time_to_datetime(df, f)

    cols_to_filter = [c 
                      for c in [
                          "ax", "ay", "az", "wx", "wy", "wz", "gFx", "gFy", "gFz", "Bx", "By", "Bz", 
                          "I", "p"
                          ] 
                          if c in df.columns]
    if cols_to_filter:
        df = butter_lowpass_filter(df, cols_to_filter, cutoff=5, fs=50, order=3)

    df = drop_unwanted_sensor_cols(df)
    sensor_df_list.append(df)

if sensor_df_list:
    sensor_df = pd.concat(sensor_df_list, ignore_index=True).sort_values("datetime").reset_index(drop = True)
else:
    sensor_df = pd.DataFrame()

sensor_df["datetime"] = pd.to_datetime(sensor_df["datetime"], errors="coerce")

if not sensor_df.empty:
    sensor_df = add_day_night(sensor_df)

labels_df = read_label_files(PROJECT_DIR)

if not audio_df.empty:
    audio_df = merge_labels(audio_df, labels_df, df_time_col="datetime")
if not sensor_df.empty:
    sensor_df = merge_labels(sensor_df, labels_df, df_time_col="datetime")

merged_df = merge_audio_sensor(audio_df, sensor_df)

train_df = merged_df.dropna(subset = ["in_out"]).copy()

train_df.to_csv("merged_dataset.csv", index=False)

candidate_cols = [c for c in [
    "ax","ay","az",
    "wx","wy","wz",
    "gFx","gFy","gFz",
    "Bx","By","Bz",
    "I","p",
    "rms"
] if c in train_df.columns]

train_df = train_df[(train_df["p"] >= 700) & (train_df["p"] <= 1200)]

print(train_df['in_out'].value_counts())
### The output for this is 3634 in's and 6400 out's
### Not too imbalanced (since F1-score and accuracy score are pretty decent)
### But I need to play it safe

results = evaluate_dual_sensors(
    train_df,
    sensor_cols = candidate_cols,
    label_col = "in_out",
    random_state = 127
)

results.to_csv("sensor_pair_results.csv", index = False)

print("Top 10 sensor pairs:")
print(results.head(10))