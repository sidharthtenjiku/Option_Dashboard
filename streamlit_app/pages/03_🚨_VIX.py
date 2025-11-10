from helpers import *
import datetime
from datetime import timezone, timedelta
import pytz
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from clickhouse_connect import get_client
from truedata.analytics import TD_analytics
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import logging

import warnings
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")
IST = timezone(timedelta(hours=5, minutes=30))

today = datetime.datetime.now(IST).date()

#! ================== FUNCTIONS ==================

def get_strike_step(symbol):
    if "BANKNIFTY" in symbol:
        return 100
    elif "NIFTY" in symbol:
        return 50
    elif "FINNIFTY" in symbol:
        return 50
    return 1


def _fetch_iv_by_expiry_today(conn,SYMBOL,td_analytics_obj):

    today = datetime.date.today().isoformat()

    #! 1. Get all rows for today and SYMBOL
    q = '''
        SELECT snap_minute_ist, expiry, strike, side, iv
        FROM option_chain_1m
        WHERE underlying = %(symbol)s
          AND toDate(snap_minute_ist) = %(today)s
        ORDER BY expiry, snap_minute_ist, strike
    '''
    rows = list(conn.query(q, parameters={"symbol": SYMBOL, "today": today}).named_results())
    if not rows:
        return pd.DataFrame(columns=["time","expiry_ts","expiry_str","tenor_tag","atm_strike","ce_iv","pe_iv","iv_avg"])
    df = pd.DataFrame(rows)
    df['expiry_str'] = df['expiry'].astype(str)
    df['expiry_ts'] = pd.to_datetime(df['expiry']).astype(int) // 10**9

    #! 2. Find the two nearest expiries (sorted ascending)
    expiry_dates = sorted(df['expiry'].unique())
    if len(expiry_dates) < 1:
        return pd.DataFrame(columns=["time","expiry_ts","expiry_str","tenor_tag","atm_strike","ce_iv","pe_iv","iv_avg"])
    near_expiry = expiry_dates[0]
    near_dt = pd.to_datetime(near_expiry)
    # Find all expiries in the next month after near expiry's month
    far_month_expiries = [e for e in expiry_dates if (pd.to_datetime(e).year > near_dt.year) or (pd.to_datetime(e).year == near_dt.year and pd.to_datetime(e).month == near_dt.month + 1)]
    if far_month_expiries:
        far_month = pd.to_datetime(far_month_expiries[0]).month
        far_year = pd.to_datetime(far_month_expiries[0]).year
        # Get all expiries in that month
        far_expiries_in_month = [e for e in expiry_dates if pd.to_datetime(e).year == far_year and pd.to_datetime(e).month == far_month]
        far_expiry = max(far_expiries_in_month)
        expiry_map = {near_expiry: 'near', far_expiry: 'far'}
    else:
        expiry_map = {near_expiry: 'near'}

    #! 3. Get spot LTP and strike step for ATM calculation
    ul_ltp = float(td_analytics_obj.get_spot_ltp(SYMBOL).LTP.iloc[0])
    strike_step = get_strike_step(SYMBOL)

    #! 4. For each expiry, for each minute, get ATM strike and IVs
    out_rows = []
    for expiry, tenor_tag in expiry_map.items():
        df_exp = df[df['expiry'] == expiry].copy()
        # For each minute
        for minute, group in df_exp.groupby(df_exp['snap_minute_ist']):
            # Find ATM strike for this minute
            strikes = group['strike'].unique()
            if len(strikes) == 0:
                continue
            # ATM = closest to ul_ltp, snapped to strike step
            atm_strike = min(strikes, key=lambda s: abs(float(s) - ul_ltp))
            # Get CE/PE IVs for ATM
            ce_iv = group[(group['strike'] == atm_strike) & (group['side'] == 'C')]['iv']
            pe_iv = group[(group['strike'] == atm_strike) & (group['side'] == 'P')]['iv']
            ce_iv = float(ce_iv.iloc[0]) if not ce_iv.empty else None
            pe_iv = float(pe_iv.iloc[0]) if not pe_iv.empty else None
            iv_avg = 0.5 * (ce_iv + pe_iv) if ce_iv is not None and pe_iv is not None else None
            out_rows.append({
                'time': pd.to_datetime(minute).strftime('%H:%M'),
                'expiry_ts': int(pd.to_datetime(expiry).timestamp()),
                'expiry_str': str(expiry),
                'tenor_tag': tenor_tag,
                'atm_strike': atm_strike,
                'ce_iv': ce_iv,
                'pe_iv': pe_iv,
                'iv_avg': iv_avg
            })
    out_df = pd.DataFrame(out_rows)
    out_df[['ce_iv', 'pe_iv', 'iv_avg']] = out_df[['ce_iv', 'pe_iv', 'iv_avg']] * 100
    return out_df.sort_values(['time','tenor_tag']).reset_index(drop=True)


def _compute_vix30(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    today = datetime.datetime.now(IST).date()
    df = df.copy()
    df["t_dt"] = pd.to_datetime(df["time"], format="%H:%M").map(lambda t: datetime.datetime.combine(today, t.time(), tzinfo=IST))
    df["expiry_dt"] = pd.to_datetime(df["expiry_ts"], unit="s", utc=True).dt.tz_convert(IST)

    out = []
    T30 = 30.0/365.0
    for t, blk in df.groupby("t_dt", sort=True):
        blk = blk.copy()
        blk["days_left"] = (blk["expiry_dt"] - t).dt.total_seconds()/86400.0
        near = blk[blk["days_left"].between(0, 30, inclusive="left")].sort_values("days_left").head(1)
        far  = blk[blk["days_left"] >= 30].sort_values("days_left").head(1)

        vix = None
        if not near.empty and not far.empty:
            iv1 = float(near["iv_avg"].iloc[0]) / 100.0
            iv2 = float(far["iv_avg"].iloc[0])  / 100.0
            T1  = float(near["days_left"].iloc[0]) / 365.0
            T2  = float(far["days_left"].iloc[0])  / 365.0
            V1, V2 = iv1*iv1*T1, iv2*iv2*T2
            w1 = (T2 - T30) / (T2 - T1) if T2 != T1 else 1.0
            V30 = w1*V1 + (1.0 - w1)*V2
            vix = 100.0*np.sqrt(V30 / T30)
        else:
            use = near if not near.empty else (far if not far.empty else None)
            if use is not None and not use.empty:
                iv = float(use["iv_avg"].iloc[0]) / 100.0
                T  = float(use["days_left"].iloc[0]) / 365.0
                if T > 0:
                    vix = 100.0 * iv * np.sqrt(T / T30)

        if vix is not None:
            out.append({"t_dt": t, "time": t.strftime("%H:%M"), "vix30": round(float(vix), 2)})

    print(
        f"{t:%H:%M} near_iv%={near['iv_avg'].iloc[0]:.2f} "
        f"near_days={near['days_left'].iloc[0]:.1f}  "
        f"far_iv%={(far['iv_avg'].iloc[0] if not far.empty else float('nan')):.2f} "
        f"far_days={(far['days_left'].iloc[0] if not far.empty else float('nan')):.1f}"
    )

    return pd.DataFrame(out).sort_values("t_dt")


def _resample_vixdf(vixdf: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    if vixdf.empty or int(interval_min) <= 1:
        return vixdf
    d = vixdf.copy()
    # ensure we have a datetime index
    if "t_dt" not in d.columns:
        today = datetime.datetime.now(IST).date()
        d["t_dt"] = pd.to_datetime(d["time"], format="%H:%M").map(
            lambda t: datetime.datetime.combine(today, t.time(), tzinfo=IST)
        )
    d = d.set_index("t_dt").sort_index()
    # last value per bucket
    out = d.resample(f"{int(interval_min)}min")[["vix30"]].last().dropna(how="all").reset_index()
    out["time"] = out["t_dt"].dt.strftime("%H:%M")
    return out[["t_dt", "time", "vix30"]]


def fetch_vix_for_indices(conn, indices, interval_min, td_analytics_obj):
    all_vixdfs = []  # to store vix dataframes for each index

    for index in indices:
        rows = _fetch_iv_by_expiry_today(conn, index, td_analytics_obj)
        if not rows.empty:
            vixdf = _compute_vix30(rows)
            vixdf = _resample_vixdf(vixdf, interval_min)

            # Add instrument column
            vixdf['instrument'] = index

            # Rename t_dt to date (optional — or keep as is)
            vixdf['date'] = pd.to_datetime(vixdf['t_dt']).dt.date
            vixdf['time'] = pd.to_datetime(vixdf['t_dt']).dt.strftime('%H:%M')

            # Keep only required columns
            vixdf = vixdf[['date', 'time', 'instrument', 'vix30']]

            all_vixdfs.append(vixdf)

    # Combine all instruments into one DataFrame
    if all_vixdfs:
        final_df = pd.concat(all_vixdfs, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame(columns=['date', 'time', 'instrument', 'vix30'])


def show_vix_params(rows, vixdf):
    if rows.empty or vixdf.empty:
        return
    last_t = vixdf["t_dt"].iloc[-1]

    today = datetime.datetime.now(IST).date()
    r = rows.copy()
    r["t_dt"] = pd.to_datetime(r["time"], format="%H:%M").map(
        lambda t: datetime.datetime.combine(today, t.time(), tzinfo=IST)
    )
    r["expiry_dt"] = pd.to_datetime(r["expiry_ts"], unit="s", utc=True).dt.tz_convert(IST)

    blk = r[r["t_dt"] == last_t].copy()
    if blk.empty: return
    blk["days_left"] = (blk["expiry_dt"] - blk["t_dt"]).dt.total_seconds()/86400.0

    near = blk[blk["days_left"].between(0, 30, inclusive="left")].sort_values("days_left").head(1)
    far  = blk[blk["days_left"] >= 30].sort_values("days_left").tail(1)  # monthly pick already upstream

    if near.empty or far.empty: return
    iv1, iv2 = float(near["iv_avg"].iloc[0]), float(far["iv_avg"].iloc[0])
    d1, d2   = float(near["days_left"].iloc[0]), float(far["days_left"].iloc[0])
    T1, T2, T30 = d1/365.0, d2/365.0, 30.0/365.0
    w1 = (T2 - T30) / (T2 - T1) if T2 != T1 else 1.0
    w2 = 1 - w1

    # Get the latest VIX 30d value
    latest_vix30 = vixdf["vix30"].iloc[-1] if not vixdf.empty and "vix30" in vixdf.columns else None
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Near IV / days", f"{iv1:.2f}% / {d1:.1f}d")
    c2.metric("Far IV / days",  f"{iv2:.2f}% / {d2:.1f}d")
    c3.metric("Weights (near/far)", f"{w1:.2f} / {w2:.2f}")
    c4.metric("VIX 30d", f"{latest_vix30:.2f}" if latest_vix30 is not None else "N/A")
    c5.metric("As On", f"{last_t.strftime('%H:%M')}")


def get_latest_vix_for_all_instruments(conn, all_instruments, td_analytics_obj):
    """Get latest VIX value for all instruments"""
    vix_data = []
    
    for instrument in all_instruments:
        try:
            rows = _fetch_iv_by_expiry_today(conn, instrument, td_analytics_obj)
            if not rows.empty:
                vixdf = _compute_vix30(rows)
                if not vixdf.empty:
                    latest_vix = vixdf.iloc[-1]['vix30']
                    vix_data.append({
                        'instrument': instrument,
                        'vix30': latest_vix
                    })
        except Exception as e:
            # Skip instruments that fail
            continue
    
    return pd.DataFrame(vix_data)

#! ================== UI ==================

st.set_page_config(page_title="🚨 VIX (Live)", layout="wide")
st.title("🚨 VIX — Live")

#! ================== Set (by default) -- universe:index | index_choice:NIFTY | stock_choice:RELIANCE ==================

ensure_session_defaults()

#! ================== ClickHouse connection check ==================

if "clickhouse_conn" not in st.session_state or st.session_state.get("clickhouse_status") != "connected":
    st.warning("Go to Home and connect to ClickHouse first.")
    st.stop()


def get_clickhouse_conn():
    return get_client(
        host="localhost",
        port=8123,
        username="ingest_w",
        password="ingest_secret",
        database="market"
    )

conn = get_clickhouse_conn()

#! ================== TrueData Analytics connection ==================
td_username = "True9030"
td_password = "vineet@9030"
td_log_level = logging.WARNING

td_analytics_obj = TD_analytics(td_username, td_password, log_level=td_log_level)

#! ================== Left Bar ==================

INDEX_LIST = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "SENSEX", "BANKEX"]

with st.sidebar:
    st.header("⚙️ Settings")

    #! Parameter -> 1 : Universe
    try:
        universe = st.segmented_control(
            "Universe",
            options=["index", "stocks"],
            format_func=lambda x: "Index" if x == "index" else "Stocks",
            key="universe"
        )
    except Exception:
        universe = st.radio(
            "Universe",
            options=["index", "stocks"],
            format_func=lambda x: "Index" if x == "index" else "Stocks",
            key="universe",
            horizontal=True
        )
    # Get all unique underlyings for OPT
    all_opt_underlyings = [
        row[0] for row in conn.query(
            "SELECT DISTINCT underlying FROM instruments WHERE instrument_type = 'OPT'"
        ).result_rows
    ]
    if universe == "index":
        symbol_choices = INDEX_LIST
        default_symbol = symbol_choices[0]
        symbol_key = "symbol"
    else:
        symbol_choices = [u for u in all_opt_underlyings if u not in INDEX_LIST]
        default_symbol = "RELIANCE" if "RELIANCE" in symbol_choices else symbol_choices[0]
        symbol_key = "symbol"

    # Only set the default index if not already set in session_state or if the value is not in the current choices
    if symbol_key not in st.session_state or st.session_state[symbol_key] not in symbol_choices:
        default_index = symbol_choices.index(default_symbol)
        st.session_state[symbol_key] = default_symbol
    else:
        default_index = symbol_choices.index(st.session_state[symbol_key])

    SYMBOL = st.selectbox(
        "Symbol",
        symbol_choices,
        index=default_index,
        key=symbol_key
    )

    #! Parameter -> 2 : interval_min 
    interval_min = st.selectbox("Interval (minutes)", [1, 3, 5, 15], index=0)


#! ================== Autorefresh Table (Based on selected time interval) ==================
refresh_ms = int(interval_min) * 60_000
st_autorefresh(interval=refresh_ms, key=f"futoi_autorefresh_{SYMBOL}_{interval_min}")

#! ------------ Caption for selected Index | Interval ------------
_univ = (st.session_state.get("universe") or "index").capitalize()
st.caption(f"**{_univ}** : `{SYMBOL or '—'}`  •  **Interval** : `{interval_min}m`")

#! ------------ If no valid symbol (e.g., stocks list empty), stop gracefully ------------
if not SYMBOL:
    st.warning("No symbol available in the selected universe.")
    st.stop()

#! ================== VIX ==================
rows = _fetch_iv_by_expiry_today(conn, SYMBOL, td_analytics_obj)

if not rows.empty:
    vixdf = _compute_vix30(rows)

    #! Display 3 tabs ie 
    #! Near IV / days | Far IV / days | Weights (near/far)
    show_vix_params(rows, vixdf)

    #! Resample to the sidebar minute interval
    vixdf = _resample_vixdf(vixdf, interval_min)

    #! Seprate table for each expiry and vix 
    expiry_groups = rows.groupby("expiry_str")
    cols = st.columns(len(expiry_groups)+ 1)
    for col, (expiry, df) in zip(cols[:-1], expiry_groups):
        with col:
            st.write(f"**Expiry: {expiry}**")
            st.write(df.drop(columns=["expiry_ts"]))
    with cols[-1]:
        st.write("**VIX 30d Table**")
        st.write(vixdf[["time", "vix30"]])

    #! Line Chart of VIX-30
    if not vixdf.empty:
        x = pd.to_datetime(vixdf["time"], format="%H:%M").map(
            lambda t: datetime.datetime.combine(today, t.time(), tzinfo=IST)
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=vixdf["vix30"], name="VIX 30d (proxy)", mode="lines"))
        fig2.update_layout(
            title=f"{SYMBOL} • 30-day VIX (proxy)",
            margin=dict(l=40,r=40,t=50,b=40),
            xaxis_title="Time", yaxis_title="Volatility (%)",
            hovermode="x unified", uirevision=f"vix30-{SYMBOL}",
        )
        fig2.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig2, use_container_width=True)

#! ================== VIX Heatmap for All Indices and Stocks ==================
# st.subheader("Latest VIX Values - All Instruments")

# # Get all available instruments
# # all_indices = [idx for idx in INDEX_LIST if idx in all_opt_underlyings]
# # all_stocks = [stock for stock in all_opt_underlyings if stock not in INDEX_LIST]
# # all_instruments = all_indices + all_stocks
# all_instruments = ["NIFTY", "BANKNIFTY"]

# if all_instruments:
#     with st.spinner("Fetching VIX values for all instruments..."):
#         vix_heatmap_data = get_latest_vix_for_all_instruments(conn, all_instruments, td_analytics_obj)

#     if not vix_heatmap_data.empty:
#         # Separate indices and stocks
#         vix_heatmap_data['type'] = vix_heatmap_data['instrument'].apply(
#             lambda x: 'Index' if x in INDEX_LIST else 'Stock'
#         )
        
#         # Sort by type (Index first) and then by VIX value (descending)
#         vix_heatmap_data = vix_heatmap_data.sort_values(['type', 'vix30'], ascending=[True, False])
        
#         # Create a matrix for heatmap: instruments as rows, single column for VIX
#         # Better structure: instruments as rows, one column showing VIX values
#         heatmap_matrix = vix_heatmap_data[['vix30']].T
#         heatmap_matrix.columns = vix_heatmap_data['instrument'].values
#         heatmap_matrix.index = ['VIX 30']
#         st.write(heatmap_matrix)

#         # Create heatmap
#         fig = px.imshow(
#             heatmap_matrix,
#             labels=dict(x="Instrument", y="", color="VIX 30"),
#             color_continuous_scale='RdYlGn_r',  # Reversed: Red = High VIX, Green = Low VIX
#             aspect="auto",
#             title='Latest VIX 30 Values Across All Instruments',
#             text_auto='.2f'
#         )
        
#         fig.update_layout(
#             height=150,
#             xaxis_title="Instrument",
#             yaxis_title=""
#         )
        
#         # Rotate x-axis labels for better readability
#         fig.update_xaxes(tickangle=-45)
        
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No VIX data available for the selected instruments.")
# else:
#     st.info("No instruments available to display.")



