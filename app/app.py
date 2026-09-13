"""
app.py — SC-ISE data loader & cleaner (fully manual column mapping).

Run locally:   streamlit run app.py

What it does, top to bottom:
  1. Upload an Excel/CSV file.
  2. Pick the sheet (Excel only — CSV skips this).
  3. Pick ONE time column + the signal (electrode) columns you want, in order.
  4. Optionally rename the signal columns (give a comma-separated list).
  5. Set sign inversion, scale factor, recorded time unit, shown time unit.
  6. Preview the cleaned table (your names, your time unit), download it as
     CSV or Excel, and take a quick look at the traces.

No auto-detection, no step detection, no calibration here on purpose: the user ran the experiment and knows their columns, so we just let them choose. The numeric work (time-unit conversion,
sign flip, scaling, NaN tidy-up) lives in ocp_tools.clean(); the plot lives in plots.py. This file is only the Streamlit layer that wires them together.
"""

## SETTING FILE PATH
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import io
import numpy as np
import pandas as pd
import streamlit as st
import re
import plotly.graph_objects as go  # only for the Figure type hint in fig_download
from io import BytesIO

import tools.ocp_tools4 as T
from tools.plots4 import (plot_ocp_time, plot_calibration, plot_corr, plot_sensor_grid, plot_all_sensors)

st.set_page_config(page_title="SC-ISE OCP / Calibration", layout="wide", page_icon="📈")

# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
# Cache the raw read so widget changes don't re-parse the whole file
@st.cache_data(show_spinner=False)
def _read_cached(file_bytes: bytes, filename: str, sheet):
    return T.read_table(io.BytesIO(file_bytes), sheet=sheet, filename=filename)

@st.cache_data(show_spinner=False)
def _clean_cached(raw, time_col, keep, channel_names, invert_sign, scale, time_in, time_out):
    return T.clean(raw, time_col, list(keep), channel_names=list(channel_names), invert_sign=invert_sign, scale=scale, time_in=time_in, time_out=time_out)

@st.cache_data(show_spinner=False)
def _steps_cached(clean_df, channels, threshold, min_channels):
    
    return T.detect_steps(clean_df, list(channels), threshold=threshold, min_channels=min_channels)

@st.cache_data(show_spinner=False)
def _organize_cached(files : dict, pattern : str):
    return T.organize_by_sensor(files, sensor_pattern=pattern)

@st.cache_data(show_spinner=False)
def _potential_cached(sensor_data : dict, exp_info : dict[list[tuple]], method : str):
    return T.calculate_potential(sensor_data, experiment_info=exp_info, method=method)

@st.cache_data(show_spinner=False)
def _calib_fig_cached(cal, ncols, use_region, min_points, r2_threshold):
    return plot_all_sensors(cal, ncols=ncols, use_region=use_region, min_points=min_points, r2_threshold=r2_threshold)

def _guess_time_col(cols: list):
    """Cheap default only: a column literally named 't'/'time', else the first."""
    low = [str(c).strip().lower() for c in cols]
    for i, c in enumerate(low):
        if c in ("t", "time", "s"): # checks if col name is either t/time/s & returns original col name
            return cols[i]
    for i, c in enumerate(low): # checks if any col name has "time" in it like "elapsed_time" & returns original col name
        if "time" in c:
            return cols[i]
    return cols[0] # if none then returns 1st col name

def fig_download(fig, stem: str, label: str):
    html = fig.to_html(include_plotlyjs="cdn").encode()
    st.download_button(f"⬇ {label} (HTML)", html, f"{stem}.html", "text/html", key=f"dl_{stem}")

# def fig_download_plt(fig, stem, label):
#     buf = BytesIO()
#     fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
#     buf.seek(0)
#     st.download_button(f"⬇ {label} (PNG)", data=buf, file_name=f"{stem}.png", mime="image/png")
        
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# converts string of name "R75, R76" to list of string ['R75', "R76"]
def _parse_names(text: str) -> list:
    return [x.strip() for x in text.split(",") if x.strip()]

def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0] if "." in filename else filename

def _guess_label(filename: str) -> str:
    """Guess a file's label (used as the E_<label> suffix). Prefers a date in the name; falls back to the 2nd underscore token (the notebook's convention)."""
    stem = filename.rsplit(".", 1)[0]
    m = re.search(r"\d{1,2}-\d{1,2}-\d{4} | \d{4}-\d{1,2}-\d{1,2}", stem)
    if m:
        return m.group() # returns matched string
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem

def _floats(text):
    """Comma-separated string -> (list of floats, list of unreadable tokens)."""
    vals, bad = [], []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            bad.append(tok)
    return vals, bad


# Streamlit drops a widget's state when the widget isn't rendered (e.g. you're in other section). Re-assigning each key marks it "still in use" so it survives.
# file_uploader / button / download_button keys CANNOT be set this way -> skip them.
SKIP = ("up_", "btn_", "dl_", "png_", "html_", "win_editor", "step_change_editor_")
for _k in list(st.session_state.keys()):
    if _k.startswith(SKIP):
        continue
    st.session_state[_k] = st.session_state[_k]


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.title("📈 SC-ISE OCP & Calibration Explorer")
#st.header("Upload a raw data exported from potentiostat")

section = st.sidebar.radio("Navigation", ["OCP Data Analysis", "Sensor Behavior"])

if section=='OCP Data Analysis':
    tabs = st.tabs(["1. Load & clean", "2. OCP vs time + steps", "3. Calibration & fit", "4. Correlation"])
    with tabs[0]:
        # =========================================================================== #
        #  1 — UPLOAD + SHEET
        # =========================================================================== #
        st.subheader("Upload your data")
        up = st.file_uploader("Upload data file here", type=["xlsx", "xls", "csv", "tsv"], key='up_ocp')
        if up is not None:
            st.session_state['ocp_file'] = {'name': up.name, 'bytes': up.getvalue()}
        blob = st.session_state.get('ocp_file')
        if blob is None:
            st.warning("Upload a file to begin."); st.stop()

        sheets = T.list_sheets(io.BytesIO(blob['bytes'])) # io.BytesIO(file_bytes) creates an in-memory file from uploaded bytes
        sheet = 0
        if sheets:
            sheet = st.selectbox("Sheet", sheets, index=0)

        raw = _read_cached(blob['bytes'], blob['name'], sheet)
        if raw.shape[1] == 0:
            st.error("That sheet has no readable columns — try another sheet.")
            st.stop()
        st.success(f"Loaded **{raw.shape[0]} rows × {raw.shape[1]} columns**.")


        # =========================================================================== #
        #  2 — COLUMN SELECTION
        # =========================================================================== #
        st.subheader("Choose your columns")
        all_cols = list(raw.columns)
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw[c])]

        # Time column stays a single-pick selectbox.
        time_col = st.selectbox("**Select time column (one for the whole table)**", all_cols, index=all_cols.index(_guess_time_col(all_cols)))
        dup_t = T.duplicate_time_cols(raw, time_col)       # extra bank time cols -> dropped
        if dup_t:
            st.caption("Dropping duplicate time column(s): " + ", ".join(map(str, dup_t)))

        all_cols = [c for c in all_cols if c != time_col and c not in dup_t] # dropping duplicated time cols of same bank

        # Signal columns: every column is shown as a chip. click a chip to select/drop it. No dropdown, no hunting; you just prune.
        st.session_state.setdefault("signal_pills", all_cols) # session_state stores variables between reruns
                                                                # setdefault creates new "signal_pills" if it doesn't exist

        st.markdown("**Select signal (electrode) columns**")
        ba, bb, _ = st.columns([1, 1, 6]) # divides the page into -> | button | button | empty space |
        if ba.button("✓ Select all", width='stretch'):
            st.session_state["signal_pills"] = all_cols
        if bb.button("✗ Clear all", width='stretch'):
            st.session_state["signal_pills"] = []

        #signal_cols = st.pills("Signal columns", all_cols, selection_mode="multi", key="signal_pills",  label_visibility="collapsed")
        signal_cols = st.multiselect("Signal columns", all_cols, default=all_cols, key="signal_pills")

        if not signal_cols:
            st.warning("Select at least one signal column.")
            st.stop()

        # # Heads-up if a non-numeric column is selected (it'll come through blank).
        # non_numeric = [c for c in signal_cols if c not in numeric_cols]
        # if non_numeric:
        #     st.caption("⚠️ Not numeric — these will come through blank, remove if unwanted: "
        #                + ", ".join(str(c) for c in non_numeric))
        st.caption(f"**{len(signal_cols)}** of {len(all_cols)} column(s) selected.")


        # =========================================================================== #
        #  3 — OPTIONAL RENAMING
        # =========================================================================== #
        st.subheader("Name your columns")
        do_rename = st.toggle("Rename the signal columns") # toggle button

        if do_rename:
            st.caption(f"Enter **{len(signal_cols)}** names separated by comma, in the same order as the columns you picked above.")
            names_text = st.text_area("New names", value=", ".join(str(c) for c in signal_cols), height=90)
            new_names = _parse_names(names_text)

            # live mapping so the order is unambiguous
            padded = (new_names + [""] * len(signal_cols))[:len(signal_cols)]
            st.dataframe(pd.DataFrame({"selected column": signal_cols, "→ new name": padded}), width='stretch', height=min(340, 45 + 35 * len(signal_cols)))

            if len(new_names) != len(signal_cols): # Check the number of names
                st.error(f"You selected {len(signal_cols)} columns but gave "
                        f"{len(new_names)} name(s). Fix the list to continue.")
                st.stop()
            if len(set(new_names)) != len(new_names): # Check for duplicate names
                st.error("Column names must be unique.")
                st.stop()
            channel_names = new_names
        else:
            channel_names = [str(c) for c in signal_cols] # keep the original headers


        # =========================================================================== #
        #  4 — SIGN / SCALE / TIME UNITS
        # =========================================================================== #
        SCALE_OPTIONS = [1.0, 1000.0, 0.001]
        SCALE_LABELS = {1.0: "none (×1)", 1000.0: "V → mV (×1000)", 0.001: "mV → V (÷1000)"}
        UNIT_ABBR = {"seconds": "s", "minutes": "min", "hours": "h"}
        Y_LABEL = {1.0: "OCP (mV)", 1000.0: "OCP (mV)", 0.001: "OCP (V)"}

        st.subheader("Sign, Scale & Time units")
        o1, o2, o3, o4 = st.columns(4)
        invert = o1.checkbox("Invert sign (−V)", value=False, help="Most setups record the negative of the OCP.")
        scale = o2.selectbox("Scale factor", SCALE_OPTIONS, format_func=lambda v: SCALE_LABELS[v])
        time_in = o3.selectbox("Time recorded in", ["seconds", "minutes", "hours"])
        time_out = o4.selectbox("Show time in", ["hours", "minutes", "seconds"])


        # =========================================================================== #
        #  5 — CLEAN + PREVIEW + DOWNLOAD
        # =========================================================================== #
        clean_df = _clean_cached(raw, time_col, signal_cols, channel_names=channel_names, invert_sign=invert, scale=scale, time_in=time_in, time_out=time_out)

        # User-facing copy: label the time column with its unit ("Time (h)").
        time_label = f"Time ({UNIT_ABBR[time_out]})"
        display_df = clean_df.rename(columns={"t": time_label})

        st.subheader("Cleaned data")
        st.dataframe(display_df.head(10), width='stretch', height=380)
        st.caption(f"{display_df.shape[0]} rows · {len(channel_names)} channels · "
                f"time {clean_df['t'].min():.4f}–{clean_df['t'].max():.4f} {UNIT_ABBR[time_out]}")

        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", display_df.to_csv(index=False).encode(), f"{_stem(up.name)}_clean.csv", "text/csv")
        d2.download_button("⬇ Download Excel", T.to_excel_bytes({"clean": display_df}), f"{_stem(up.name)}_clean.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # encode() => converts this string into bytes so Streamlit can send it as a file
        # text/csv => tells browser that it's a CSV file
        # T.to_excel_bytes({"clean": display_df} => creates an Excel file & adds a worksheet called clean


    # =========================================================================== #
    #  TAB 2 — OCP vs TIME + STEP DETECTION
    # =========================================================================== #
    with tabs[1]:
        st.subheader("Plot channels")
        st.markdown("**Select which channels to plot**")
    
        # A separate pick for which channels to plot, independent of the ones you cleaned. Reset it to "all" whenever
        # channel set changes (e.g. after a rename or re-picking signal columns); otherwise keep the pick across reruns.
        if st.session_state.get("_plot_src") != tuple(channel_names):
            st.session_state["_plot_src"] = tuple(channel_names)
            st.session_state["plot_channels"] = list(channel_names)
    
        pa, pb, _ = st.columns([1, 1, 6])  # | button | button | empty space |
        if pa.button("✓ Plot all", width='stretch', key="btn_plot_all"):
            st.session_state["plot_channels"] = channel_names
        if pb.button("✗ Plot none", width='stretch', key="btn_plot_none"):
            st.session_state["plot_channels"] = []
    
        # chips to select/deselect channels for the plot (same widget as the column picker)
        #plot_cols = st.pills("plot channels", channel_names, selection_mode="multi", key="plot_channels", label_visibility="collapsed")
        plot_cols = st.multiselect("Channels/Electrodes", channel_names, default=channel_names, key="plot_channels")
        overlay = st.toggle("Overlay all channels (off = grid of small plots)", value=False, key="overlay_tab1")
    
        if plot_cols:
            fig = plot_ocp_time(clean_df, plot_cols, overlay=overlay, time_label=time_label, y_label=Y_LABEL[scale])
            st.plotly_chart(fig, width='stretch')
            fig_download(fig, "ocp_vs_time", "OCP vs time")
        else:
            st.info("Select at least one channel to plot.")  
        
        st.divider()
        st.subheader("Concentration steps")
        mode = st.radio("How do you want to set the switch times?", ["Auto-detect from the data", "Enter them manually"], horizontal=True, key=f'step_change_radio')
        t_max = float(clean_df["t"].max())
        st.write(t_max)
        if mode == "Auto-detect from the data":
            s1, s2 = st.columns(2)
            thr = s1.slider("Jump threshold (mV)", 1.0, 50.0, 7.0, 0.5, help="A step is flagged where the OCP jumps more than this threshold between consecutive points.", key=f'step_change_slider2')
            n_ch = len(plot_cols)
            max_ch = max(2, n_ch)
            if "step_change_slider2_" in st.session_state:
                st.session_state["step_change_slider2_"] = min(st.session_state["step_change_slider2_"], max_ch)
            min_ch = s2.slider("Min channels jumping together", 1, max_ch, max(1, n_ch // 2), help="A real switch shows up on many channels at once.", key=f'step_change_slider2_')
            steps = _steps_cached(clean_df, channel_names, threshold=thr, min_channels=min_ch)
            boundaries = sorted(round(float(t), 6) for t in steps["t"].tolist())
            det_sig  = tuple(boundaries)
            if st.session_state.get(f"_det_sig") != det_sig:
                st.session_state[f"_det_sig"] = det_sig
                st.session_state[f'step_change_times'] = ", ".join(f"{t:g}" for t in boundaries)
            st.caption(f"Auto-detected **{len(boundaries)}** switch times.")
            times_txt = st.text_input("Switch times. Remove unwanted if there are any. [NOTE that this does not include star & end time these are only concentration change time step]",
                                        key=f'step_change_times')
        else:
            # txt = st.text_input("Switch times (comma-separated, in shown time unit)", value="")
            # boundaries = [float(x) for x in txt.split(",") if x.strip()]
            # st.success(f"{len(boundaries)} switch times → {len(boundaries)+1} windows.")
            times_txt = st.text_input("Switch times (comma-separated, in shown time unit)", value="", key=f'step_change_text1')
            
        t_vals, t_bad = _floats(times_txt)
        boundaries = sorted(t for t in t_vals if 0 < t < t_max)   # keep only in-range switches
        if t_bad:
            st.warning("Ignoring unreadable switch-time entries: " + ", ".join(t_bad))
        # concentration sequence -> build/refresh the editable windows table
        nwin = len(boundaries) + 1
        st.success(f"**{len(boundaries)}** switch times → **{nwin}** windows.")
        
        conc_txt = st.text_input(f"Enter **{nwin}** concentrations (M), comma-separated, in window order (e.g. 1e-8, 1e-7, 1e-6, ...)", value="", key=f'step_change_text2')
        c_vals, c_bad = _floats(conc_txt)
        ready = len(c_vals) == nwin
        if c_bad:
            st.warning("Ignoring unreadable concentration entries: " + ", ".join(c_bad))
        if not conc_txt.strip():
            st.info(f"Enter {nwin} concentrations to build the windows table.")
        elif not ready:
            st.error(f"Got {len(c_vals)} concentrations but need {nwin}. Fix the concentrations, or switch times above.")
        
        auto_windows = T.build_windows(boundaries, t_max, c_vals if ready else [])
        st.session_state["windows_ocp"] = auto_windows

        if ready:
            st.markdown("##### Windows &nbsp;·&nbsp; *final*")
            st.caption("This table shows start time, end time for each concentration if there are any changes make them above this'll update automatically.")
            # win = st.data_editor(st.session_state["windows_ocp"], key="win_editor", num_rows="dynamic", width='stretch', height=300,
            #                 column_config={"start": st.column_config.NumberColumn("Start", format="%.4f"), "end": st.column_config.NumberColumn("End", format="%.4f"),
            #                                 "concentration": st.column_config.NumberColumn("Conc (M)", format="%.2e")})
            st.dataframe(auto_windows, width='stretch', height=min(360, 45 + 35 * nwin), hide_index=True, column_config={"start": st.column_config.NumberColumn("Start", format="%.4f"),
                                       "end": st.column_config.NumberColumn("End", format="%.4f"), "concentration": st.column_config.NumberColumn("Conc (M)", format="%.2e")})
        #st.session_state["windows_ocp"] = win
        
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", auto_windows.to_csv(index=False).encode(), f"{_stem(up.name)}_time_windows.csv", "text/csv", key=f'dl_step_change_csv')
        d2.download_button("⬇ Download Excel", T.to_excel_bytes({"windows": auto_windows}), f"{_stem(up.name)}_time_windows.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f'dl_step_change_excel')
        
        
    # =========================================================================== #
    #  TAB 3 — CALIBRATION & FIT
    # =========================================================================== #
    with tabs[2]:
        if st.session_state.get("windows_ocp") is None or st.session_state["windows_ocp"].empty:
            st.info("Set up concentration windows in tab 2 first.")
            st.stop()

        cmap = T.make_channel_map(channel_names)
        win = st.session_state["windows_ocp"]

        st.subheader("Calibration curve & linear fit")
        f1, f2, f3, f4 = st.columns(4)
        method = f1.radio("OCP per window", ["last", "avg"], format_func=lambda m: {"last": "Last point", "avg": "Average"}[m], horizontal=True)
        use_region = f2.toggle("Auto-find best linear region", help="Brute-force the contiguous concentration range with the highest R².")
        # prefer = f3.radio("Prefer method to select best line", ["r2","width"], format_func=lambda m: {"r2": "R2 score", "width": "Points"}[m], horizontal=True, disabled=not use_region,
        #                   help=("**Points:** Gives line having maximum number points such that r2 score > min r2.  It prefers no. of points over r2 score.\n\n"
        #                         "**R2 Score:** Gives line having maximum r2 score with points > min points. It prefers r2 score over no. of points."))
        r2_thr = f3.slider("Min R² for best region", 0.80, 0.999, 0.98, 0.001, disabled=not use_region)
        min_points = f4.slider("Min points for best region", 2, len(win), 4, 1, disabled=not use_region)

        seg, calib = T.build_calibration(clean_df, win, method=method)
        #st.write(seg)

        # exclude points by log-concentration
        all_logc = sorted(calib["logC"].dropna().unique())
        excl = st.multiselect("Exclude points (by log₁₀ concentration) — e.g. drop the flat low-conc tail", options=[round(x, 2) for x in all_logc], default=[])

        fit = T.fit_calibration(calib, exclude_logc=excl, use_best_region=use_region, r2_threshold=r2_thr, min_points=min_points)

        fig = plot_calibration(calib if not excl else calib[~calib["logC"].round(2).isin(excl)], fit)
        st.plotly_chart(fig, width='stretch')
        fig_download(fig, "calibration_curve", "Calibration curve")

        st.markdown("##### Sensitivity table")
        show_fit = fit.rename(columns={"slope_mV_per_dec": "Sensitivity (mV/dec)", "intercept_mV": "Intercept (mV)", "r2": "R²", "n_points": "Points"})
        st.dataframe(show_fit.style.format({"Sensitivity (mV/dec)": "{:.2f}", "Intercept (mV)": "{:.2f}", "R²": "{:.4f}"}), width='stretch')

        d3, d4 = st.columns(2)
        d3.download_button("⬇ Download CSV", show_fit.to_csv(index=False).encode(), f"{_stem(up.name)}_fit.csv", "text/csv", key='dl_fit_csv')
        d4.download_button("⬇ Download Excel", T.to_excel_bytes({"fit_line_data": show_fit}), f"{_stem(up.name)}_fit.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        , key='dl_fit_excel')
        
    #     st.markdown("##### Per-electrode summary")
    #     exp1, exp2 = st.columns(2)
    #     use_exp = exp1.checkbox("Flag against an expected Nernstian slope")
    #     exp_slope = exp2.number_input("Expected slope (mV/dec)", value=-29.5, disabled=not use_exp, help="e.g. ≈ −29.5 for a divalent cation (Ca²⁺), ≈ −59 for a monovalent ion.")
    #     summary = T.build_summary(fit, r2_min=(r2_thr if use_region else 0.98), expected_slope=(exp_slope if use_exp else None))
    #     st.dataframe(summary, width='stretch')

    #     st.divider()
    #     st.markdown("##### Downloads")
    #     d1, d2, d3 = st.columns(3)
    #     d1.download_button("⬇ Fit results (CSV)", fit.to_csv(index=False).encode(), "fit_results.csv", "text/csv")
    #     d2.download_button("⬇ Calibration long-form (CSV)", calib.to_csv(index=False).encode(), "calibration_points.csv", "text/csv")
    #     d3.download_button("⬇ Full report (Excel)",
    #         T.to_excel_bytes({"summary": summary, "fit": fit, "calibration_points": calib, "windows": win, "channel_map": cmap}),
    #         "calibration_report.xlsx",
    #         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        
    # =========================================================================== #
    #  TAB 4 — CORRELATION
    # =========================================================================== #
    with tabs[3]:
        cmap =  T.make_channel_map(channel_names)
        chans = channel_names
        st.subheader("Channel-to-channel correlation")
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True, help=("**Pearson:** Measures linear relationships. Best when the data follows a straight-line trend.\n\n"
                                                                                    "**Spearman:** Measures monotonic relationships using ranks. More robust to outliers and non-linear trends."))
        corr = T.correlation(clean_df, chans, method=method)
        fig = plot_corr(corr)
        st.plotly_chart(fig, width='stretch')
        fig_download(fig, "correlation", "Correlation matrix")
        st.download_button("⬇ Correlation matrix (CSV)", corr.to_csv().encode(), "correlation.csv", "text/csv")
    
    

elif section=='Sensor Behavior':
    tab5, tab6 = st.tabs(['OCP vs time', 'Calibration curve'])
    # =========================================================================== #
    #  TAB 5 — SENSOR BEHAVIOR OVER TIME
    # =========================================================================== #
    with tab5:
        st.subheader("Sensor behaviour across dates - OCP vs time")
        st.caption("Upload one file per date. For each file, pick its time column and OCP columns and rename them to sensor IDs (R75, R76,…). Each sensor's OCP-vs-time is then plotted across every" 
                "date it appears in.")

        ups = st.file_uploader("Upload data files (one per date)", type=["xlsx", "xls", "csv", "tsv"], accept_multiple_files=True, key="up_multi")
        if ups:
            st.session_state['sensor_files'] = [{'name': f.name, 'bytes': f.getvalue()} for f in ups]
        blobs = st.session_state.get('sensor_files')
        if blobs is None:
            st.warning("Upload one or more file to begin."); st.stop()

        files, labels, experiment_info = {}, [], {}
        for k, u in enumerate(blobs):
            raw = _read_cached(u['bytes'], u['name'], 0) # reads file
            with st.expander(f"{u['name']} · {raw.shape[0]} rows × {raw.shape[1]} cols", expanded=(len(ups) == 1)): # creates a drop expander for each uploaded file
                
                # READING, DROPPING DUPLICATED TIME COLS & RENAMING COLUMNS
                cols = list(raw.columns) # list of cols for each uploade file
                c_lab, c_time = st.columns(2)
                label = c_lab.text_input("Date / label (column suffix)", value=_guess_label(u['name']), key=f"lab_{k}") # text_input for column
                time_col = c_time.selectbox("Time column", cols, index=cols.index(_guess_time_col(cols)), key=f"time_{k}") # time col selection box

                dup_t = T.duplicate_time_cols(raw, time_col)       # extra bank time cols -> dropped
                if dup_t:
                    st.caption("Dropping duplicate time column(s): " + ", ".join(map(str, dup_t)))

                candidates = [c for c in cols if c != time_col and c not in dup_t] # dropping duplicated time cols of same bank
                keep = st.multiselect("OCP (sensor) columns", candidates, default=candidates, key=f"keep_{k}") # user will select required OCP cols
                if not keep:
                    st.warning("Pick at least one OCP column.")
                    continue
                
                st.caption(f"Enter **{len(keep)}** names separated by comma, in the same order as the columns you picked above.")
                sig = tuple(keep)
                if st.session_state.get(f"_keep_sig_{k}") != sig:
                    st.session_state[f"_keep_sig_{k}"] = sig
                    st.session_state[f"text_name{k}"] = ", ".join(str(c) for c in keep)
                names_text = st.text_area("New names as per convection like R75, R76,...", value=", ".join(str(c) for c in keep), height=90, key=f'text_name{k}')
                new_names = _parse_names(names_text)

                # live mapping so the order is unambiguous
                new_candidates = (new_names + [""] * len(keep))[:len(keep)]
                st.dataframe(pd.DataFrame({"selected column": keep, "→ new name": new_candidates}), width='stretch', height=min(340, 45 + 35 * len(keep)))
                
                # SCALING, INVERTING SIGN AND TIME_IN & TIME_OUT UNITS
                SCALE_OPTIONS = [1.0, 1000.0, 0.001]
                SCALE_LABELS = {1.0: "none (×1)", 1000.0: "V → mV (×1000)", 0.001: "mV → V (÷1000)"}
                UNIT_ABBR = {"seconds": "s", "minutes": "min", "hours": "h"}
                Y_LABEL = {1.0: "OCP (mV)", 1000.0: "OCP (mV)", 0.001: "OCP (V)"}
                
                st.subheader("Sign, Scale & Time units")
                o1, o2, o3, o4 = st.columns(4)
                invert = o1.checkbox("Invert sign (−V)", value=False, help="Most setups record the negative of the OCP.", key=f'invert_cleaned_{k}')
                scale = o2.selectbox("Scale factor", SCALE_OPTIONS, format_func=lambda v: SCALE_LABELS[v], key=f'scale_cleaned_{k}')
                time_in = o3.selectbox("Time recorded in", ["seconds", "minutes", "hours"], key=f'time_in_cleaned_{k}')
                time_out = o4.selectbox("Show time in", ["hours", "minutes", "seconds"], key=f'time_out_cleaned_{k}')
                if len(new_names) != len(keep):
                    st.error(f"In {u['name']}: {len(keep)} columns selected but {len(new_names)} name(s) given.")
                    continue
                if len(set(new_names)) != len(new_names):
                    st.error(f"In {u['name']}: column names must be unique.")
                    continue
                clean_df = _clean_cached(raw, time_col, keep, channel_names=new_names, invert_sign=invert, scale=scale, time_in=time_in, time_out=time_out)
                time_label = f"Time ({UNIT_ABBR[time_out]})"
                display_df = clean_df.rename(columns={"t": time_label})
            
                # CALCULATING CONC CHANGE STEPS
                # CONCENTRATION STEPS
                st.subheader("Concentration steps")
                mode = st.radio("How do you want to set the switch times?", ["Auto-detect from the data", "Enter them manually"], horizontal=True, key=f'step_change_radio_{k}')
                t_max = float(clean_df["t"].max())
                # STEP 1 — switch times in an editable box (auto-detect just seeds it)
                if mode == "Auto-detect from the data":
                    s1, s2 = st.columns(2)
                    thr = s1.slider("Jump threshold (mV)", 1.0, 50.0, 7.0, 0.5, help="A step is flagged where the OCP jumps more than this between consecutive points.",
                                    key=f'step_change_slider1_{k}')
                    n_ch = len(keep)
                    min_ch = s2.slider("Min channels jumping together", 1, max(2, n_ch), max(1, n_ch // 2), help="A real switch shows up on many channels at once.",
                                       key=f'step_change_slider2_{k}_{n_ch}')
                    steps = _steps_cached(clean_df, new_names, threshold=thr, min_channels=min_ch)
                    detected = sorted(round(float(t), 6) for t in steps["t"].tolist())
                    # re-seed the box only when detection actually changes, so manual edits survive
                    det_sig  = tuple(detected)
                    if st.session_state.get(f"_det_sig_{k}") != det_sig:
                        st.session_state[f"_det_sig_{k}"] = det_sig
                        st.session_state[f'step_change_times_{k}'] = ", ".join(f"{t:.6f}" for t in detected)
                    st.caption(f"Auto-detected **{len(detected)}** switch times.")
                    times_txt = st.text_input("Switch times. Remove unwanted if there are any. [NOTE that this does not include star & end time these are only concentration change time step]",
                                              key=f'step_change_times_{k}')
                else:
                    times_txt = st.text_input("Switch times (comma-separated, in shown time unit)", value="", key=f'step_change_text1_{k}')
                    
                t_vals, t_bad = _floats(times_txt)
                boundaries = sorted(t for t in t_vals if 0 < t < t_max)   # keep only in-range switches
                if t_bad:
                    st.warning("Ignoring unreadable switch-time entries: " + ", ".join(t_bad))
                nwin = len(boundaries) + 1
                st.success(f"**{len(boundaries)}** switch times → **{nwin}** windows.")

                conc_txt = st.text_input(f"Enter **{nwin}** concentrations (M), comma-separated, in window order (e.g. 1e-8, 1e-7, 1e-6, ...)", value="", key=f'step_change_text2_{k}')
                c_vals, c_bad = _floats(conc_txt)
                ready = len(c_vals) == nwin
                if c_bad:
                    st.warning("Ignoring unreadable concentration entries: " + ", ".join(c_bad))
                if not conc_txt.strip():
                    st.info(f"Enter {nwin} concentrations to build the windows table.")
                elif not ready:
                    st.error(f"Got {len(c_vals)} concentrations but need {nwin}. Fix the concentrations, or switch times above.")

                windows = T.build_windows(boundaries, t_max, c_vals if ready else [])
                st.session_state[f"windows_{label}"] = windows
                experiment_info[label] = [tuple(windows.iloc[i]) for i in range(len(windows))]

                if ready:
                    st.markdown("##### Windows &nbsp;·&nbsp; *final*")
                    st.caption("This table shows start time, end time for each concentration if there are any changes make them above this'll update automatically.")
                    st.dataframe(windows, width='stretch', height=min(360, 45 + 35 * nwin), hide_index=True, column_config={"start": st.column_config.NumberColumn("Start", format="%.4f"),
                                       "end": st.column_config.NumberColumn("End", format="%.4f"), "concentration": st.column_config.NumberColumn("Conc (M)", format="%.2e")})

                    d_1, d_2 = st.columns(2)
                    d_1.download_button("⬇ Download CSV", windows.to_csv(index=False).encode(), "time_windows.csv", "text/csv", key=f'dl_step_change_csv_{k}')
                    d_2.download_button("⬇ Download Excel", T.to_excel_bytes({"windows": windows}), "time_windows.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f'dl_step_change_excel_{k}')          
            
            channel_names = [str(c) for c in keep] # keep the original headers
                    
            df = pd.DataFrame({"t": pd.to_numeric(display_df[time_label], errors="coerce")})
            for src, dst in zip(keep, channel_names):
                df[dst] = pd.to_numeric(display_df[src], errors="coerce")
                files[label] = df
                labels.append(label) # files => key -> date, values -> df on that date with scaling, renaming cols, etc
            
        if not files:
            st.info("Configure at least one file above.")
            st.stop()
        # if len(set(labels)) != len(labels):
        #     st.error("Two files share a label — make them unique. If you have 2 files with same date then replace Date/Label(column suffix value) as for example 12-16-2025(1) & 12-16-2025(2) to parse this error")
        #     st.stop()

        #st.write(experiment_info)
        sensor_data = _organize_cached(files, pattern=None)  # col 0 = time, rest = sensors
        if not sensor_data:
            st.warning("No sensor columns to combine — check your selections.")
            st.stop()
        st.success(f"Combined **{len(sensor_data)}** sensor(s) across **{len(files)}** date(s).")
        #st.write(sensor_data)
        
        gcols = st.slider("Columns in grid", 2, 8, max(2, min(4, len(sensor_data))))
        sensor_over_time = plot_sensor_grid(sensor_data, ncols=gcols, time_label=f'Time ({UNIT_ABBR[time_out]})', y_label="OCP (mV)")
        st.plotly_chart(sensor_over_time, width='stretch')
        fig_download(sensor_over_time, "sensor_over_time", "sensor_over_time")
        
        st.subheader("Sensor data")
        st.caption("Select sensor to get the combined data across date of that sensor.")
        sensor = st.selectbox("Sensor Name", options=sensor_data.keys())
        st.dataframe(sensor_data[sensor].head(10), width='stretch', height=380)
        st.caption(f"{sensor_data[sensor].shape[0]} rows · {sensor_data[sensor].shape[1]//2} dates")

        d6, d7 = st.columns(2)
        d6.download_button("⬇ Download CSV", sensor_data[sensor].to_csv(index=False).encode(), f"combined_data_{sensor}.csv", "text/csv", key='dl_csv_cleaned_data')
        d7.download_button("⬇ Download Excel", T.to_excel_bytes({"clean": sensor_data[sensor]}), f"combined_data_{sensor}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='dl_excel_cleaned_data')
        
    with tab6:
        st.subheader("Calibration curve & linear fit")
        
        f1, f2, f3, f4 = st.columns(4)
        method = f1.radio("OCP per window", ["last", "average"], format_func=lambda m: {"last": "Last point", "average": "Average"}[m], horizontal=True)
        use_region = f2.toggle("Auto-find best linear region", help="Brute-force the contiguous concentration range with the highest R².")
        prefer = f3.radio("Prefer method to select best line", ["r2","width"], format_func=lambda m: {"r2": "R2 score", "width": "Points"}[m], horizontal=True, disabled=not use_region,
                          help=("**Points:** Gives line having maximum number points such that r2 score > min r2.  It prefers no. of points over r2 score.\n\n"
                                "**R2 Score:** Gives line having maximum r2 score with points > min points. It prefers r2 score over no. of points."))
        r2_thr = f4.slider("Min R² for best region", 0.80, 0.999, 0.98, 0.001, disabled=not use_region)
        min_points = f4.slider("Min points for best region", 2, nwin, nwin//2, 1, disabled=not use_region)

        calib = _potential_cached(sensor_data, exp_info=experiment_info, method=method)
        #st.write(calib)
        # # exclude points by log-concentration
        # all_logc = sorted(calib["logC"].dropna().unique())
        #excl = st.multiselect("Exclude points (by log₁₀ concentration) — e.g. drop the flat low-conc tail", options=[round(x, 2) for x in all_logc], default=[])
        #fit = T.fit_calibration(calib, exclude_logc=excl, use_best_region=use_region, r2_threshold=r2_thr, min_points=min_points)

        gcols = st.slider("Columns in grid", 2, 8, max(2, min(4, len(sensor_data))), key='sensor_behavior_cols')
        fig = plot_all_sensors(calib, use_region=use_region, min_points=min_points, r2_threshold=r2_thr, ncols=gcols, prefer=prefer)
        st.plotly_chart(fig, width='stretch')
        fig_download(fig, "calibragion_curve", 'Calibration curve')
        

### CONVERT ALL PLOTLY FUNCTION TO MATPLOTLIB - Done
### CACHE SO THAT ON SWITCHING RESULTS ARE STILL THERE
### OPTIMISE WH0LE APP SO THAT IT TAKES LESS LOAD
### DEPLOY