"""
ocp_tools.py
------------
Pure data-processing logic for the SC-ISE loader. No Streamlit and no plotting
here on purpose: everything in this file can be imported and unit-tested on its
own.

Pipeline, in order:
    list_sheets()   -> sheet names for an uploaded xlsx (or [] for csv)
    read_table()    -> raw DataFrame from an uploaded xlsx/csv, header sniffed
    clean()         -> tidy DataFrame: one 't' column + one column per channel, with sign / scale / time-unit applied
    to_excel_bytes()-> write one or more DataFrames to an xlsx in memory
    detect_format() -> a *suggested* column mapping (time col, signal cols, banks, duplicate-time columns).
                       The UI shows this pre-filled and lets the user override it.
    detect_steps() -> candidate concentration-switch times from the data.
    build_windows() -> turn boundary times + a concentration list into (start, end, concentration) windows.
    build_calibration() -> avg or last OCP per channel per window (long form).
    fit_calibration()   -> per-channel linear fit (slope = sensitivity, R2).
    correlation()       -> channel-channel correlation matrix.
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import re
from scipy.stats import linregress


# --------------------------------------------------------------------------- #
#  Reading
# --------------------------------------------------------------------------- #
# Sheet names for an xlsx, or [] for a csv/tsv.
def list_sheets(file) -> list[str]:
    try:
        return pd.ExcelFile(file).sheet_names
    except Exception:
        return []

# A row is a header if most of its cells are non-numeric text
def _row_is_header(row) -> bool:
    vals = row.tolist()
    non_numeric = 0
    for v in vals:
        if isinstance(v, str): # checks if v is string or not => bool
            try:
                float(v)
            except ValueError:
                non_numeric += 1
    return non_numeric >= max(1, len(vals) // 2)

def read_table(file, sheet=0, filename: str = "") -> pd.DataFrame:
    """
    Read an uploaded file into a DataFrame with *no* assumption about headers. We read header=None first, then sniff
    whether the first row is a header (mostly non-numeric) or already data. Returns a DataFrame whose columns are strings
    (real header names, or 'col0', 'col1', ... if none). Duplicate header names are disambiguated as 'E', 'E.1', 'E.2',...
    """
    is_csv = str(filename).lower().endswith((".csv", ".tsv", ".txt")) # checks if file is excel or csv type => bool
    sep = "\t" if str(filename).lower().endswith(".tsv") else None # setting separator to "tab" if it's tsv file

    if is_csv:
        try:
            raw = pd.read_csv(file, header=None, sep=sep, engine="python") # reading csv file with enocding utf-8 (default)
        except UnicodeDecodeError:
            file.seek(0) # rewind before retrying
            raw = pd.read_csv(file, header=None, sep=sep, engine="python", encoding="utf-16") # reading csv file with enocding utf-16
    else:
        raw = pd.read_excel(file, header=None, sheet_name=sheet) # reading excel file

    if _row_is_header(raw.iloc[0]):
        header = raw.iloc[0].astype(str).tolist()
        raw = raw.iloc[1:].reset_index(drop=True)

        seen: dict[str, int] = {}# disambiguate duplicate header names
        cols = []
        for h in header:
            h = h.strip() if isinstance(h, str) else str(h)
            if h in seen:
                seen[h] += 1
                h = f"{h}.{seen[h]}"
            else:
                seen[h] = 0
            cols.append(h)
        raw.columns = cols
    else:
        raw.columns = [f"col{i}" for i in range(raw.shape[1])]

    # Best-effort numeric coercion per column: convert when it doesn't destroy
    # real values, else keep the column as text.
    for c in raw.columns:
        conv = pd.to_numeric(raw[c], errors="coerce")
        if conv.notna().sum() == raw[c].notna().sum():
            raw[c] = conv
    return raw


# --------------------------------------------------------------------------- #
#  Cleaning
# --------------------------------------------------------------------------- #
TIME_UNIT_TO_HOURS = {"seconds": 1 / 3600, "minutes": 1 / 60, "hours": 1.0}

# Build the tidy frame: a 't' column + 1 col per channel/electrode (converted from time_in -> time_out)
def clean(raw: pd.DataFrame, time_col, signal_cols: list, channel_names: list | None = None, invert_sign: bool = True,
          scale: float = 1.0, time_in: str = "seconds", time_out: str = "hours") -> pd.DataFrame:
    if channel_names is None:
        channel_names = [f"ch{i+1}" for i in range(len(signal_cols))]
    if len(channel_names) != len(signal_cols):
        raise ValueError("channel_names and signal_cols length mismatch")

    out = pd.DataFrame()
    t = pd.to_numeric(raw[time_col], errors="coerce")
    t_hours = t * TIME_UNIT_TO_HOURS[time_in]
    out["t"] = t_hours / TIME_UNIT_TO_HOURS[time_out]

    for name, col in zip(channel_names, signal_cols):
        v = pd.to_numeric(raw[col], errors="coerce")
        if invert_sign:
            v = -v
        out[name] = v * scale

    out = out.dropna(subset=["t"]).reset_index(drop=True)
    # drop rows that are entirely NaN across channels (trailing bank padding)
    out = out.dropna(subset=channel_names, how="all").reset_index(drop=True)
    return out

def make_channel_map(channel_names) -> pd.DataFrame:
    return pd.DataFrame({"channel": list(channel_names), "active": [True] * len(channel_names)})
    
def active_channels(channel_map: pd.DataFrame) -> list:
    return channel_map.loc[channel_map["active"], "channel"].tolist()

def duplicate_time_cols(df, time_col, tol=1e-9):
    """Columns (other than time_col) whose values equal the chosen time column — the repeated per-bank time columns in a raw multi-channel export."""
    ref = pd.to_numeric(df[time_col], errors="coerce")
    out = []
    for c in df.columns:
        if c == time_col:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if (np.array_equal(v.notna().values, ref.notna().values) and len(v.dropna()) == len(ref.dropna()) and np.allclose(v.dropna().values, ref.dropna().values, atol=tol, rtol=0)):
            out.append(c)
    return out

def organize_by_sensor(files: dict, sensor_pattern: str = r"R\d+") -> dict:
    sensor_names = set()
    for df in files.values():
        if sensor_pattern is None:                         # <-- new branch
            sensor_names.update(df.columns[1:])            # col 0 = time, rest = sensors
        else:
            sensor_names.update(c for c in df.columns if re.fullmatch(sensor_pattern, str(c)))

    sensor_data = {}
    for sensor in sorted(sensor_names, key=str):
        pieces = []
        for label, df in files.items():
            if sensor not in df.columns:
                continue
            time_name = str(df.columns[0])
            pieces.append(pd.DataFrame({f"{time_name}_{label}": df.iloc[:, 0].values,f"E_{label}": df[sensor].values,}))
        sensor_data[sensor] = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    return sensor_data

# --------------------------------------------------------------------------- #
#  Step (concentration-switch) detection
# --------------------------------------------------------------------------- #
def detect_steps(clean_df: pd.DataFrame, channels: list, threshold: float = 7.0, min_channels: int = 10, time_col: str = "t",) -> pd.DataFrame:
    """Find times where many channels jump at once (a concentration switch). Returns a DataFrame of candidate boundary
    times with how many channels jumped there."""
    counts: dict[float, int] = {} # key => time, value => no of channels having change > threshold at that time
    for ch in channels:
        diff = clean_df[ch].diff()
        idx = np.where(diff.abs() > threshold)[0]
        for i in idx:
            tval = round(float(clean_df[time_col].iloc[i]), 6)
            counts[tval] = counts.get(tval, 0) + 1 
            # dictionary.get(key, default_value) => gives val of key if key exists o/w default val

    if len(counts) == 0:
        return pd.DataFrame(columns=[time_col, "n_channels"]) # empty df => so tune threshold or min_channels accordingly
    out = (pd.DataFrame({time_col: list(counts.keys()), "n_channels": list(counts.values())})
           .query("n_channels >= @min_channels").sort_values(time_col).reset_index(drop=True))
    return out

def build_windows(boundaries: list, t_max: float, concentrations: list) -> pd.DataFrame:
    """
    Turn interior boundary times into (start, end, concentration) windows.
    boundaries: interior switch times (not including 0 or t_max).
    concentrations: one per window (len == len(boundaries)+1).
    """
    edges = [0.0] + sorted(float(b) for b in boundaries) + [float(t_max)] # all time step from 0 to t_max
    rows = []
    for i in range(len(edges) - 1):
        conc = concentrations[i] if i < len(concentrations) else np.nan
        rows.append({"start": edges[i], "end": edges[i + 1], "concentration": conc})
    return pd.DataFrame(rows)

def calculate_potential(sensor_data, experiment_info, method='last', edge_drop=1): # function to get calibration curve data
    calibration_data = {}
    for el, df in sensor_data.items():
        dfs = []
        for i in range(0, df.shape[1], 2):
            t = df.iloc[:, i].values
            E = df.iloc[:, i + 1].values
            date = df.columns[i + 1].replace("E_", "")
            wins = experiment_info[date]
            logC, potential = [], []
            
            for j, (start, end, conc) in enumerate(wins):
                last_win = (j == len(wins) - 1)
                mask = ((t >= start) & (t < end) if last_win else (t < end))
                idx = np.flatnonzero(mask & ~np.isnan(E))
                if edge_drop and not last_win and idx.size > edge_drop:
                    idx = idx[:-edge_drop]          # never straddle the switch
                logC.append(np.log10(conc) if (conc and conc > 0) else np.nan)
                if idx.size == 0:
                    potential.append(np.nan)
                elif method == "last":
                    potential.append(E[idx[-1]])
                elif method == "average":
                    potential.append(E[idx].mean())
                else:
                    raise ValueError("method must be 'average' or 'last'")
            
            key = "Avg_OCP" if method == "average" else "Last_OCP"
            dfs.append(pd.DataFrame({f"logC_{date}": logC, f"{key}_{date}": potential}))
            #dfs.append(temp)
        calibration_data[el] = pd.concat(dfs, axis=1)
    return calibration_data


# --------------------------------------------------------------------------- #
#  Calibration table
# --------------------------------------------------------------------------- #
# Long-form calibration table: one row per (channel, window)

def build_calibration(clean_df, windows, method="avg", time_col="t", guard=0.0, edge_drop=1):
    """Windows are half-open [start, end). Returns (segments, calib_df)."""
    rows, segments = [], {}
    t = clean_df[time_col]

    for ch in clean_df.columns[1:]:
        for i, w in windows.iterrows():
            mask = (t >= w["start"] + guard) & (t < w["end"] - guard)
            idx = clean_df.index[mask]
            if edge_drop and len(idx) > edge_drop:
                idx = idx[:-edge_drop]

            seg = clean_df.loc[idx, ch].dropna()
            segments[f"{ch}_{i}"] = seg          # still a Series, downstream-safe

            if seg.empty:
                ocp, t_used = np.nan, np.nan
            else:
                ocp = seg.iloc[-1] if method == "last" else seg.mean()
                t_used = clean_df.loc[seg.index[-1], time_col]

            conc = w["concentration"]
            rows.append({"channel": ch, "window": i, "conc_M": conc, "logC": np.log10(conc) if (conc and conc > 0) else np.nan, "OCP_mV": ocp, "t_used": t_used, "n_pts": len(seg)})

    return segments, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Linear fitting
# --------------------------------------------------------------------------- #
def fit_one(sub: pd.DataFrame, x="logC", y="OCP_mV"):
    sub = sub.dropna(subset=[x, y]).sort_values(x)
    if len(sub) < 2:
        return None
    s, b, r, _, _ = linregress(sub[x], sub[y])
    return {"slope": s, "intercept": b, "r2": r ** 2, "n": len(sub), "logC_min": sub[x].min(), "logC_max": sub[x].max()}

# Brute-force the contiguous sub-range with the best R2 (notebook logic)
def best_linear_region(sub: pd.DataFrame, min_points=4, r2_threshold=0.98, x="logC", y="OCP_mV"):
    sub = sub.dropna(subset=[x, y]).sort_values(x).reset_index(drop=True)
    n = len(sub)
    best, best_r2 = None, -np.inf
    for start in range(n):
        for end in range(start + min_points, n + 1):
            subset = sub.iloc[start:end]
            s, b, r, _, _ = linregress(subset[x], subset[y])
            if r ** 2 > best_r2 and r ** 2 >= r2_threshold:
                best, best_r2 = subset, r ** 2
    return best

# Per-channel linear fit -> table of slope (sensitivity), intercept, R2
def fit_calibration(calib_df: pd.DataFrame, exclude_logc: list | None = None, use_best_region: bool = False,
                    min_points: int = 4, r2_threshold: float = 0.98,) -> pd.DataFrame:
    exclude_logc = set(round(v, 6) for v in (exclude_logc or []))
    rows = []
    for ch, g in calib_df.groupby("channel", sort=False):
        #electrode = g["electrode"].iloc[0]
        sub = g.copy()
        if exclude_logc:
            sub = sub[~sub["logC"].round(6).isin(exclude_logc)]
        if use_best_region:
            region = best_linear_region(sub, min_points=min_points, r2_threshold=r2_threshold)
            sub = region if region is not None else sub.iloc[0:0]
        fit = fit_one(sub)

        if fit is None:
            rows.append({"channel": ch, "slope_mV_per_dec": np.nan, "intercept_mV": np.nan, "r2": np.nan, "n_points": 0})
            continue
        
        rows.append({"channel": ch, "slope_mV_per_dec": fit["slope"], "intercept_mV": fit["intercept"], "r2": fit["r2"], "n_points": fit["n"], "fit_min": fit["logC_min"],
                     "fit_max": fit["logC_max"]})
    return pd.DataFrame(rows)


def build_summary(fit_df: pd.DataFrame, r2_min=0.98, expected_slope=None, slope_tol=10.0) -> pd.DataFrame:
    """ Per-electrode summary with simple quality flags. `expected_slope` is optional (e.g. -29.5 mV/dec for a divalent
    cation like Ca2+); if given, flags electrodes whose sensitivity is far from Nernstian.
    """
    out = fit_df.copy()
    out["r2_ok"] = out["r2"] >= r2_min
    if expected_slope is not None:
        out["slope_ok"] = (out["slope_mV_per_dec"] - expected_slope).abs() <= slope_tol
        out["quality"] = np.where(out["r2_ok"] & out["slope_ok"], "good",
                          np.where(out["r2_ok"] | out["slope_ok"], "check", "poor"))
    else:
        out["quality"] = np.where(out["r2_ok"], "good", "check")
    return out


# --------------------------------------------------------------------------- #
#  Correlation
# --------------------------------------------------------------------------- #
def correlation(clean_df: pd.DataFrame, channels: list, method="pearson") -> pd.DataFrame:
    return clean_df[channels].corr(method=method)


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write one or more DataFrames to a single xlsx in memory."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
