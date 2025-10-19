import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import librosa
import matplotlib
from sklearn.metrics import f1_score, accuracy_score


sns.set_theme(style="whitegrid")

PROJECT_DIR = "project_data"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
matplotlib.use('Agg') 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_audio_file(file_path, lowcut=20, highcut=20000):
    y, sr = librosa.load(file_path, sr=None, mono=True)

    n_sec = len(y) // sr
    y_trunc = y[: n_sec * sr] if n_sec > 0 else y

    if n_sec > 0:
        y_2d = y_trunc.reshape(n_sec, sr)
        raw_rms = np.sqrt(np.mean(y_2d ** 2, axis=1))
    else:
        raw_rms = np.array([np.sqrt(np.mean(y_trunc ** 2))])

    filtered = butter_bandpass_filter(y, lowcut, highcut, sr)

    if n_sec > 0:
        filtered_2d = filtered[: n_sec * sr].reshape(n_sec, sr)
        filtered_rms = np.sqrt(np.mean(filtered_2d ** 2, axis=1))
    else:
        filtered_rms = np.array([np.sqrt(np.mean(filtered ** 2))])

    df = pd.DataFrame({
        "time_sec": np.arange(len(raw_rms)),
        "raw_rms": raw_rms,
        "filtered_rms": filtered_rms,
        "file": os.path.basename(file_path)
    })
    return df

audio_files = glob.glob(os.path.join(PROJECT_DIR, "*.wav"))
audio_dfs = Parallel(n_jobs=-1)(delayed(process_audio_file)(f) for f in audio_files)
audio_df_all = pd.concat(audio_dfs, ignore_index=True)

for f in audio_df_all["file"].unique():
    df_plot = audio_df_all[audio_df_all["file"] == f]
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["time_sec"], df_plot["raw_rms"], 'b.', alpha = 0.3, label = "Raw RMS")
    plt.plot(df_plot["time_sec"], df_plot["filtered_rms"], 'r-', label = "Filtered RMS")
    plt.xlabel("Seconds")
    plt.ylabel("RMS")
    plt.title(f"Audio RMS: {f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"audio_rms_{f}.svg"))
    plt.close()

def butter_lowpass_filter(data, cutoff=5, fs=50, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

sensor_cols = ["ax","ay","az","wx","wy","wz","gFx","gFy","gFz","Bx","By","Bz","I","p"]
sensor_files = glob.glob(os.path.join(PROJECT_DIR, "*.csv"))
print(sensor_files)

def process_sensor_file(f):
    df = pd.read_csv(f)
    df = df[(df["p"] >= 700) & (df["p"] <= 1200)]  
    df_filtered = df.copy()
    for c in sensor_cols:
        if c in df.columns:
            df_filtered[c] = butter_lowpass_filter(df[c].values)
    return df, df_filtered, f

sensor_processed = Parallel(n_jobs = -1)(delayed(process_sensor_file)(f) for f in sensor_files)

for raw_df, filtered_df, fname in sensor_processed:
    base_fname = os.path.basename(fname)
    for c in sensor_cols:
        if c in raw_df.columns:
            plt.figure(figsize=(12,4))
            plt.plot(raw_df.index, raw_df[c], 'b.', alpha = 0.3 , label = "Raw")
            plt.plot(filtered_df.index, filtered_df[c], 'r-', label = "Filtered")
            plt.xlabel("Index")
            plt.ylabel(c)
            plt.title(f"Sensor: {c} ({base_fname})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{c}_{base_fname}.svg"))
            plt.close()

#-----------------------------------------------------------------------------------------------
merged_df = pd.read_csv("merged_dataset.csv")
results_df = pd.read_csv("sensor_pair_results.csv")

best_sensor_pair = results_df.iloc[0]["dual_sensors"].split("+")
feature_cols = []
SENSOR_MAP = {"accelerometer": ["ax", "ay", "az"], 
              "gyroscope": ["wx", "wy", "wz"], 
              "gravity": ["gFx", "gFy", "gFz"], 
              "magnetometer": ["Bx", "By", "Bz"], 
              "light meter": ["I"], "barometer": ["p"], "audio": ["rms"] }
for s in best_sensor_pair:
    feature_cols += SENSOR_MAP[s]

X = merged_df[feature_cols]
y = merged_df["in_out"].map({"in": "inside", "out": "outside"})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=127)

rf = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 30, 
                            n_jobs = -1, random_state = 127)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

pca = PCA(n_components=2, random_state=127)
X_test_pca = pca.fit_transform(X_test)

pca_df = pd.DataFrame({
    "PC1": X_test_pca[:, 0],
    "PC2": X_test_pca[:, 1],
    "truth": y_test.values,
    "prediction": y_pred
})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="truth",        
    style="prediction", 
    palette="Set1"
)
plt.title("PCA Projection of Classification Results")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "classification_pca.svg"))
plt.close()

misclassified = X_test[y_test != y_pred].copy()
misclassified["truth"] = y_test[y_test != y_pred]
misclassified["prediction"] = y_pred[y_test != y_pred]
misclassified.to_csv(os.path.join(OUTPUT_DIR, "misclassified_points.csv"), index = False)

# ------------------------------------------------------------------------------------------
merged_df = merged_df[(merged_df["p"] >= 700) & (merged_df["p"] <= 1200)]
all_features = merged_df.columns.values
to_exclude = np.array(["datetime", "time", "time_of_day_sec", "in_out"])
all_features = np.setdiff1d(all_features, to_exclude)

X = merged_df[all_features]
y = merged_df["in_out"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 127)

clf = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 30, 
                             n_jobs = -1, random_state = 127, class_weight = 'balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = "macro")

print("All sensors model:")
print("Accuracy:", accuracy)
print("F1 score:", f1)

results = pd.DataFrame([{
    "model": "all_sensors",
    "features_used": all_features,
    "n_samples": len(y),
    "accuracy": accuracy,
    "f1_score": f1
}])

results.to_csv("all_sensor_results.csv", mode = "a", header = False, index = False)
