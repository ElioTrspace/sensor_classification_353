import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

SENSOR_MAP = {
    "ax": "accelerometer", "ay": "accelerometer", "az": "accelerometer",
    "wx": "gyroscope", "wy": "gyroscope", "wz": "gyroscope",
    "gFx": "gravity", "gFy": "gravity", "gFz": "gravity",
    "Bx": "magnetometer", "By": "magnetometer", "Bz": "magnetometer",
    "I": "light meter", "p": "barometer",
    "rms": "audio"
}

def _prep_xy(df, feature_cols, label_col = "in_out"):
    """
    Prepare X and y. Remove NaN/inf, filter labels to 'in'/'out'.
    """
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, label_col]
    mask = y.isin(["in", "out"])
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y.map({"in": "inside", "out": "outside"})

def evaluate_dual_sensors(df, sensor_cols, label_col = "in_out", test_size = 0.25, 
                          random_state = 127):
    """
    Evaluate all combinations of dual sensors.
    For each dual sensor pair, train/test a RandomForest using all features belonging to the sensors.
    """
    results = []

    sensor_types_present = set(SENSOR_MAP[c] for c in sensor_cols if c in SENSOR_MAP)
    dual_sensor_pairs = list(itertools.combinations(sensor_types_present, 2))

    for sensor_pair in dual_sensor_pairs:
        features = [col for col in sensor_cols if SENSOR_MAP.get(col) in sensor_pair]

        if "light meter" in sensor_pair and "day_or_night_day" in df.columns:
            features += ["day_or_night_day", "day_or_night_night"]

        if len(features) == 0:
            continue

        X, y = _prep_xy(df, features, label_col = label_col)
        if len(y.unique()) < 2:
            continue 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, stratify = y, random_state = random_state
        )

        rf = RandomForestClassifier(
            n_estimators = 200,
            min_samples_leaf = 30,
            n_jobs = -1,
            random_state = random_state,
            class_weight = 'balanced'
        )
        
        rf.fit(X_train, y_train)
        acc = rf.score(X_test, y_test)
        f1 = f1_score(y_test, rf.predict(X_test), average = "macro")

        results.append({
            "dual_sensors": "+".join(sensor_pair),
            "features_used": features,
            "n_samples": len(y),
            "accuracy": acc,
            "f1_score": f1
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)