"""
No Streamlit here so these can be tested on their own; app.py imports them and just calls st.plotly_chart on the result.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tools import ocp_tools4 as T
from scipy.stats import linregress

def _date_of(colname, prefix=None):
    """'E_12-08-2025' or 'logC_12-08-2025' -> '12-08-2025'."""
    s = str(colname)
    if prefix and s.startswith(prefix):
        return s[len(prefix):]
    return s.split("_", 1)[1] if "_" in s else s

def _vspace(nrows, want=0.08):
    """Plotly requires vertical_spacing <= 1/(nrows-1); clamp so tall grids don't crash."""
    if nrows < 2:
        return 0
    return min(want, 0.9 / (nrows - 1))


PALETTE = px.colors.qualitative.Dark24

def plot_ocp_time(df, channels, overlay=False, boundaries=None, time_label="Time (h)", y_label="OCP (mV)"):
    if not len(channels):
        return go.Figure()
    if overlay:
        fig = go.Figure()
        for i, ch in enumerate(channels):
            fig.add_scatter(x=df["t"], y=df[ch], mode="lines", name=str(ch), line=dict(width=1.2, color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(xaxis_title=time_label, yaxis_title=y_label, height=520, legend=dict(font=dict(size=9)), margin=dict(l=40, r=10, t=30, b=40))
        for b in (boundaries or []):
            fig.add_vline(x=b, line=dict(color="rgba(120,120,120,0.5)", dash="dot"))
    else:
        n = len(channels)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(ch) for ch in channels], vertical_spacing=0.08, horizontal_spacing=0.05)
        for i, ch in enumerate(channels):
            r, c = i // ncols + 1, i % ncols + 1
            fig.add_scatter(x=df["t"], y=df[ch], mode="lines", row=r, col=c, showlegend=False, line=dict(width=1, color=PALETTE[i % len(PALETTE)]),)
            for b in (boundaries or []):
                fig.add_vline(x=b, line=dict(color="rgba(120,120,120,0.4)", dash="dot"), row=r, col=c)
        fig.update_layout(height=max(300, 230 * nrows), margin=dict(l=30, r=10, t=40, b=30))
        fig.update_xaxes(title_text=time_label)
        fig.update_yaxes(title_text=y_label)
        fig.update_annotations(font_size=11)
    return fig

def plot_calibration(calib, fit, x_label='logC', y_label="OCP (mV)"):
    fitmap = {row["channel"]: row for _, row in fit.iterrows()}
    channels = calib['channel'].unique()
    n = len(channels)
    if not n:
        return go.Figure()
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(ch) for ch in channels], vertical_spacing=0.08, horizontal_spacing=0.05)
    for i, (ch, g) in enumerate(calib.groupby("channel", sort=False)):
        g = g.dropna(subset=["logC", "OCP_mV"]).sort_values("logC")
        if g.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        r, c = i // ncols + 1, i % ncols + 1
        fig.add_scatter(x=g["logC"], y=g["OCP_mV"], mode="markers", row=r, col=c, showlegend=False, marker=dict(size=7, color=color))
        f = fitmap.get(ch)
        if f is not None and not np.isnan(f["slope_mV_per_dec"]):
            xs = np.array([f["fit_min"], f["fit_max"]])
            ys = f["intercept_mV"] + f["slope_mV_per_dec"] * xs
            fig.add_scatter(x=xs, y=ys, mode="lines", row=r, col=c, line=dict(color=color, dash="dash", width=1), showlegend=False, hoverinfo="skip")
    fig.update_layout(xaxis_title="log\u2081\u2080[concentration] (M)", yaxis_title="OCP (mV)", height=max(360, 260 * nrows), legend=dict(font=dict(size=9)), margin=dict(l=50, r=10, t=30, b=40))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_annotations(font_size=11)
    return fig


def plot_corr(corr):
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_layout(height=max(420, 26 * len(corr) + 120), margin=dict(l=40, r=10, t=30, b=40))
    fig.update_traces(textfont_size=8)
    return fig


## SENSOR WISE
def plot_sensor_grid(sensor_data, ncols=None, time_label="Time (s)", y_label="OCP (mV)"):
    """Grid of sensors — one subplot each, overlaying that sensor's OCP-vs-time for every date it appears in. Values are converted V -> mV via mv_scale (raw -V -> +mV by default, like 
    notebook); set mv_scale=1 if your data is already mV."""
    sensors = list(sensor_data.keys())
    n = len(sensors)
    if not n:
        return None, None
    
    # one fixed colour per date, shared across subplots, shown once in the legend
    dates = []
    for df in sensor_data.values():
        for j in range(1, df.shape[1], 2):
            d = _date_of(df.columns[j])
            if d not in dates:
                dates.append(d)
    color = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(dates)}

    ncols = ncols or int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    max_spacing = 1 / (nrows - 1) if nrows > 1 else 0
    vertical_spacing = min(0.03, max_spacing * 0.9)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sensors, vertical_spacing=vertical_spacing, horizontal_spacing=0.05)
    seen = set()
    for i, sensor in enumerate(sensors):
        df = sensor_data[sensor]
        r, c = i // ncols + 1, i % ncols + 1
        for j in range(0, df.shape[1], 2):
            tcol, ecol = df.columns[j], df.columns[j + 1]
            sub = df[[tcol, ecol]].dropna()
            date = _date_of(ecol)
            fig.add_scatter(x=sub[tcol], y=sub[ecol], mode="lines", name=date, line=dict(width=1, color=color[date]), legendgroup=date, showlegend=(date not in seen), row=r, col=c)
            seen.add(date)
    fig.update_layout(height=max(320, 260 * nrows), legend=dict(font=dict(size=9)), margin=dict(l=40, r=10, t=40, b=40))
    fig.update_xaxes(title_text=time_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_annotations(font_size=11)
    return fig
    

def _best_region(x, y, min_points=4, r2_threshold=0.98, prefer="r2"):
    """x, y already NaN-free and sorted by x. Returns (xs, ys, slope, int, r2) or None."""
    n = len(x)
    best, best_key = None, None
    for s in range(n):
        for e in range(s + min_points, n + 1):
            xs, ys = x[s:e], y[s:e] # subset of x & y from start to end
            if xs.max() - xs.min() == 0:
                continue
            slope, intercept, r, _, _ = linregress(xs, ys) # r => pearson correlation coefficient
            r2 = r * r # r2 score
            if r2 < r2_threshold:
                continue
            key = (e - s, r2) if prefer == "width" else (r2, e - s)
            if best_key is None or key > best_key:
                best_key, best = key, (xs, ys, slope, intercept, r2)
    return best

def plot_sensor_calibration(fig, df, row, col, seen, min_points=4, r2_threshold=0.98, prefer="r2", show_slope=True, use_region=True):
    """Overlay every date in one sensor's (logC_<date>, OCP_<date>) frame on `ax`."""
    pairs = [(df.columns[i], df.columns[i + 1]) for i in range(0, df.shape[1], 2)] # pairs of logC & Avg/Last_OCP datewise
    plotted = 0
    for k, (xcol, ycol) in enumerate(pairs):
        date = _date_of(xcol, prefix="logC_")   # date
        sub = df[[xcol, ycol]].dropna() # Removing NaNs per-date mostly trailing ones as duration is not same for each date
        if sub.empty:
            continue
        x = sub[xcol].to_numpy(float) # logC values & not pd series
        y = sub[ycol].to_numpy(float) # Avg/Last_OCP values & not pd series
        order = np.argsort(x) # returns indices of sorted array eg x=[30,10,20] => o/p=[3,1,2]
        x, y = x[order], y[order] # sorts x & y based on sorted order of x
        show = date not in seen
        seen.add(date)
        c = PALETTE[k % len(PALETTE)]
        fig.add_scatter(x=x, y=y, mode="lines+markers", name=date, legendgroup=date, showlegend=show, line=dict(color=c, width=0.8, dash="dot"), marker=dict(size=5, symbol="circle-open", color=c),
                        row=row, col=col) # all points
        
        if use_region:
            best = _best_region(x, y, min_points, r2_threshold, prefer) # x_best, y_best, slope, intercept, r2
            if best is None:
                continue
            xs, ys, slope, intercept, r2 = best
            fig.add_scatter(x=xs, y=ys, mode="markers", legendgroup=date, showlegend=False, marker=dict(size=8, color=c), row=row, col=col) # region points filled with color and greater size
            xfit = xs
        else:
            if len(np.unique(x)) < 2:
                return f"{date}: <2 distinct x"; continue
            slope, intercept, rr, _, _ = linregress(x, y)
            r2 = rr * rr
            xfit = x
                    
        lbl = f"{date}: Slope={slope:.1f}, R²={r2:.4f}" if show_slope else date
        fig.add_scatter(x=xfit, y=intercept + slope * xfit, mode="lines", legendgroup=date, showlegend=False, line=dict(color=c, width=2, dash="dash"),
                        hovertemplate=lbl + "<extra></extra>", name=lbl, row=row, col=col)
        plotted += 1
    return plotted

def plot_all_sensors(cal_dict, prefer, ncols=4, figsize=(5, 5), **kw):
    """
    Grid of sensors (default) — each subplot overlays that sensor's dates.
    per_figure=True -> one separate, larger figure per sensor instead.
    """
    sensors = list(cal_dict.keys())
    seen = set()
    nrows = int(np.ceil(len(sensors) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sensors, vertical_spacing=0.05, horizontal_spacing=0.06)
    for i, s in enumerate(sensors):
        row = i // ncols + 1
        col = i % ncols + 1
        plot_sensor_calibration(fig, cal_dict[s], row=row, col=col, seen=seen, prefer=prefer, **kw)
    fig.update_layout(height = 300 * nrows)
    fig.update_xaxes(title_text="logC")
    fig.update_yaxes(title_text="OCP (mV)")
    fig.update_annotations(font_size=11)
    return fig