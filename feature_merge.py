import pandas as pd

def merge_audio_sensor(audio_df, sensor_df):
    """
    Nearest-time merge on absolute datetime.
    I only keep necessary columns from audio to avoid duplicate 'time' confusion.
    Output has a single 'in_out' column:
      - prefer sensor label if present, else audio label.
    """
    a = audio_df.sort_values("datetime")
    s = sensor_df.sort_values("datetime")

    a_keep = ["datetime", "rms", "in_out"]
    a_keep = [c for c in a_keep if c in a.columns]
    s_keep = [c for c in s.columns if c != "in_out"] + (["in_out"] if "in_out" in s.columns else [])

    merged = pd.merge_asof(
        a[a_keep],
        s[s_keep],
        on = "datetime",
        direction = "nearest"
    )

    if "in_out_x" in merged.columns or "in_out_y" in merged.columns:
        merged["in_out"] = merged.get("in_out_y", pd.Series(index=merged.index)).combine_first(
            merged.get("in_out_x", pd.Series(index=merged.index))
        )
        merged = merged.drop(columns=[c for c in ["in_out_x", "in_out_y"] if c in merged.columns])

    unnamed = [c for c in merged.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        merged = merged.drop(columns=unnamed)

    return merged