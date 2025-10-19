import pandas as pd
import glob
import os

def _parse_label_file_date_from_name(fname):
    """
    I know:
      - file without "(2)" is Aug 4 (2025-08-04)
      - file with "(2)" is Aug 5 (2025-08-05)
    """
    base = os.path.basename(fname)
    # Default to Aug 4 if "(2)" not present; Aug 5 if present
    if "(2)" in base:
        date_str = "2025-08-05"
    else:
        date_str = "2025-08-04"
    return pd.Timestamp(date_str).date()


def read_label_files(label_folder):
    """
    Read all '*Timestamp*.txt' files. Expected rows like: "19:04:10, in, 1", then:
      - parse time (HH:MM:SS),
      - attach a DATE inferred from filename rule above,
      - build absolute datetime,
      - keep 'in_out' as the label,
      - also compute 'time_of_day_sec' for robust merges.
    Returns columns: ['datetime','in_out','time_of_day_sec']
    """
    paths = glob.glob(os.path.join(label_folder, "*Timestamp*.txt"))
    dfs = []
    for p in paths:
        df = pd.read_csv(p, header=None, names=["time", "in_out", "ignore"])
        df = df[["time", "in_out"]].copy()
        df["in_out"] = df["in_out"].astype(str).str.strip().str.lower()
        df["time"] = pd.to_datetime(df["time"].astype(str), errors="coerce").dt.time
        df = df.dropna(subset=["time"])

        label_date = _parse_label_file_date_from_name(p)
        df["datetime"] = df["time"].apply(lambda t: pd.Timestamp.combine(label_date, t))
        df["time_of_day_sec"] = df["datetime"].dt.hour * 3600 + df["datetime"].dt.minute * 60 + df["datetime"].dt.second

        dfs.append(df[["datetime", "in_out", "time_of_day_sec"]])

    if not dfs:
        return pd.DataFrame(columns=["datetime", "in_out", "time_of_day_sec"])

    labels = pd.concat(dfs, ignore_index = True).sort_values("datetime").reset_index(drop=True)
    return labels


def merge_labels(df, labels_df, df_time_col="datetime", tolerance_seconds=120):
    """
    Attach labels to a dataframe with absolute datetimes by matching on time-of-day (seconds from midnight).
    This avoids date mismatches and lets multiple days use the same label schedule.
    """
    if df_time_col not in df.columns:
        raise KeyError(f"Expected '{df_time_col}' in df")

    tmp = df.copy()
    tmp["time_of_day_sec"] = (
        pd.to_datetime(tmp[df_time_col]).dt.hour * 3600
        + pd.to_datetime(tmp[df_time_col]).dt.minute * 60
        + pd.to_datetime(tmp[df_time_col]).dt.second
    )

    lab = labels_df[["time_of_day_sec", "in_out"]].dropna().sort_values("time_of_day_sec")
    tmp = tmp.sort_values("time_of_day_sec")

    merged = pd.merge_asof(
        tmp, lab,
        on="time_of_day_sec",
        direction="nearest",
        tolerance=tolerance_seconds
    )

    return merged.sort_index()