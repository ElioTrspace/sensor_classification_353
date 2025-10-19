import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_audio_rms(raw_df, filtered_df, save_path="audio_rms.svg"):
    """
    Plot raw and filtered RMS over time.
    raw_df, filtered_df: must have columns 'datetime' and 'rms'
    """
    plt.figure(figsize=(12,6))
    plt.plot(raw_df['datetime'], raw_df['rms'], label='Raw RMS', alpha=0.5)
    plt.plot(filtered_df['datetime'], filtered_df['rms'], label='Filtered RMS', alpha=0.9)
    plt.xlabel("Time")
    plt.ylabel("RMS")
    plt.title("Audio RMS: Raw vs Filtered")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_sensors(raw_sensor_df, filtered_sensor_df, sensor_cols, save_prefix="sensor"):
    """
    Plot raw vs filtered sensor values per sensor.
    """
    for col in sensor_cols:
        if col not in raw_sensor_df.columns or col not in filtered_sensor_df.columns:
            continue
        plt.figure(figsize=(12,4))
        plt.plot(raw_sensor_df['datetime'], raw_sensor_df[col], alpha=0.5, label=f"{col} Raw")
        plt.plot(filtered_sensor_df['datetime'], filtered_sensor_df[col], alpha=0.9, label=f"{col} Filtered")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.title(f"Sensor {col}: Raw vs Filtered")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{col}.svg")
        plt.show()

def plot_misclassifications(y_true, y_pred, dual_sensor_pairs, save_path="mistakes_by_dual_sensor.svg"):
    """
    y_true, y_pred: pd.Series or list
    dual_sensor_pairs: list or pd.Series mapping each sample to its dual sensor
    """
    df_eval = pd.DataFrame({
        "truth": y_true,
        "prediction": y_pred,
        "dual_sensor_pair": dual_sensor_pairs
    })
    mistakes = df_eval[df_eval['truth'] != df_eval['prediction']]
    mistake_counts = mistakes['dual_sensor_pair'].value_counts()

    plt.figure(figsize=(10,6))
    sns.barplot(x=mistake_counts.index, y=mistake_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Dual Sensor Pair")
    plt.ylabel("Number of Misclassifications")
    plt.title("Model Mistakes by Dual Sensor Pair")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return df_eval, mistakes