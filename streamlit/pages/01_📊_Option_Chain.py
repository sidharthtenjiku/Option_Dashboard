from helpers import *
import datetime
import pytz
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from clickhouse_connect import get_client
from truedata.analytics import TD_analytics

import warnings
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

#! MASTER FUNCTIONS

def add_oi_trigger(df, threshold_percent=10, threshold_absolute=1000):
    """
    CORRECTED OI Trigger column based on significant OI changes
    Now properly handles mixed scripts/strikes/expiries by grouping correctly
    
    Args:
        df: DataFrame with option data
        threshold_percent: Percentage threshold for OI change trigger (default: 10%)
        threshold_absolute: Absolute threshold for OI change trigger (default: 1000 contracts)
    
    Returns:
        DataFrame with oi_trigger column added
    """
    
    # Initialize oi_trigger column
    df['oi_trigger'] = 'NORMAL'
    
    # Calculate absolute OI change
    df['oi_change_abs'] = df['oi_change'].abs()
    
    # Apply trigger logic
    # Trigger if either percentage change > threshold OR absolute change > threshold
    trigger_conditions = (
        (df['oi_change_perc'].abs() > threshold_percent) |  # More than 10% change
        (df['oi_change_abs'] > threshold_absolute)          # More than 1000 contracts change
    )
    
    # Set trigger types based on direction
    df.loc[trigger_conditions & (df['oi_change'] > 0), 'oi_trigger'] = 'OI_BUILDUP'
    df.loc[trigger_conditions & (df['oi_change'] < 0), 'oi_trigger'] = 'OI_UNWIND'
    
    # Add intensity levels
    high_intensity = (
        (df['oi_change_perc'].abs() > threshold_percent * 2) |  # More than 20% change
        (df['oi_change_abs'] > threshold_absolute * 2)           # More than 2000 contracts change
    )
    
    df.loc[high_intensity & (df['oi_change'] > 0), 'oi_trigger'] = 'HIGH_OI_BUILDUP'
    df.loc[high_intensity & (df['oi_change'] < 0), 'oi_trigger'] = 'HIGH_OI_UNWIND'
    
    # Clean up temporary column
    df = df.drop('oi_change_abs', axis=1)
    
    return df

def add_oi_trap(df, high_oi_threshold=0.6, significant_oi_change=5):
    """
    CORRECTED oi_trap column based on Call/Put Writers Trap logic
    Now properly handles mixed scripts/strikes/expiries by grouping correctly
    
    OI Trap Types:
    - Call Writers Trap: High call OI + OI decreasing (writers trapped)
    - Put Writers Trap: High put OI + OI decreasing (writers trapped)
    - Call Writers Trap-Pentry: High call OI + OI increasing (new bullish positions)
    - Put Writers Trap-Pentry: High put OI + OI increasing (new bearish positions)
    - Call Writers Trap-Pexit: High call OI + OI increasing (unwinding)
    - Put Writers Trap-Pexit: High put OI + OI increasing (unwinding)
    - No Trap: No significant trap detected
    
    Args:
        df: DataFrame with option data
        high_oi_threshold: Threshold for high OI (default: 0.6) (60th percentile within the same underlying)
        significant_oi_change: Threshold for significant OI change (default: 5%)

    Returns:
        DataFrame with oi_trap column added
    """
    
    # Initialize oi_trap column
    df['oi_trap'] = 'No Trap'
    
    # Sort by timestamp for proper analysis
    df = df.sort_values(['underlying', 'expiry', 'strike', 'side', 'timestamp_ist']).copy()
    
    # CORRECTED: Calculate OI percentiles per underlying and trading session
    # This ensures we compare OI within the same underlying, not across different scripts
    df['oi_percentile'] = df.groupby(
        ['underlying', 'expiry', 'strike', 'side', df['timestamp_ist'].dt.date]
    )['oi'].transform(lambda x: x.rank(pct=True))
    
    print(f"📊 CORRECTED Trap Analysis:")
    print(f"High OI threshold: {high_oi_threshold} (per underlying)")
    print(f"Significant OI change: {significant_oi_change}%")
    print(f"OI percentile range: {df['oi_percentile'].min():.3f} to {df['oi_percentile'].max():.3f}")
    print(f"OI change % range: {df['oi_change_perc'].min():.2f}% to {df['oi_change_perc'].max():.2f}%")
    
    #! Call Writers Trap Logic
    call_mask = df['side'] == 'C'
    
    #! Call Writers Trap: High call OI + OI decreasing (writers trapped)
    call_trap_conditions = (
        call_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] < 0) &  # OI decreasing (writers trapped)
        (df['oi_change_perc'] < -significant_oi_change)  # Significant decrease
    )
    df.loc[call_trap_conditions, 'oi_trap'] = 'Call Writers Trap'
    
    #! Call Writers Trap-Pentry: High call OI + OI increasing (new bullish positions)
    call_entry_conditions = (
        call_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] > 0) &  # OI increasing (new positions)
        (df['oi_change_perc'] > significant_oi_change)  # Significant increase
    )
    df.loc[call_entry_conditions, 'oi_trap'] = 'Call Writers Trap-Pentry'
    
    #! Call Writers Trap-Pexit: High call OI + OI increasing (unwinding)
    call_exit_conditions = (
        call_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] > 0) &  # OI increasing (unwinding)
        (df['oi_change_perc'] > significant_oi_change) &  # Significant increase
        (df['oi_change_perc'] < significant_oi_change * 2)  # But not too extreme
    )
    df.loc[call_exit_conditions, 'oi_trap'] = 'Call Writers Trap-Pexit'
    
    #! Put Writers Trap Logic
    put_mask = df['side'] == 'P'
    
    #! Put Writers Trap: High put OI + OI decreasing (writers trapped)
    put_trap_conditions = (
        put_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] < 0) &  # OI decreasing (writers trapped)
        (df['oi_change_perc'] < -significant_oi_change)  # Significant decrease
    )
    df.loc[put_trap_conditions, 'oi_trap'] = 'Put Writers Trap'
    
    #! Put Writers Trap-Pentry: High put OI + OI increasing (new bearish positions)
    put_entry_conditions = (
        put_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] > 0) &  # OI increasing (new positions)
        (df['oi_change_perc'] > significant_oi_change)  # Significant increase
    )
    df.loc[put_entry_conditions, 'oi_trap'] = 'Put Writers Trap-Pentry'
    
    #! Put Writers Trap-Pexit: High put OI + OI increasing (unwinding)
    put_exit_conditions = (
        put_mask &
        (df['oi_percentile'] > high_oi_threshold) &
        (df['oi_change'] > 0) &  # OI increasing (unwinding)
        (df['oi_change_perc'] > significant_oi_change) &  # Significant increase
        (df['oi_change_perc'] < significant_oi_change * 2)  # But not too extreme
    )
    df.loc[put_exit_conditions, 'oi_trap'] = 'Put Writers Trap-Pexit'
    
    #! Show oi_trap detection statistics
    oi_trap_counts = df['oi_trap'].value_counts()
    print(f"\n🎯 Traps Detection Summary:")
    for oi_trap, count in oi_trap_counts.items():
        print(f"{oi_trap}: {count} records")
    
    #! Clean up temporary columns
    df = df.drop(['oi_percentile'], axis=1)
    
    return df

def add_delta_change(df, delta_change_threshold=0.01, delta_velocity_threshold=0.001):
    """
    Add Δ-Delta (Intraday Change in Delta) Feature
    
    This tracks how Delta changes over time for each unique option contract
    Each unique combination of (underlying, strike, expiry, side) is treated as a separate script
    
    Args:
        df: DataFrame with option data
        delta_change_threshold: Threshold for Delta change (default: 0.01) (1% Delta change)
        delta_velocity_threshold: Threshold for Delta velocity (default: 0.001) (0.1% per minute)
    
    Returns:
        DataFrame with Δ-Delta columns added
    """
    
    # Sort by unique option contract and timestamp
    df = df.sort_values(['underlying', 'strike', 'expiry', 'side', 'timestamp_ist']).copy()
    
    # Calculate Δ-Delta for each unique option contract
    df['delta_change'] = df.groupby(['underlying', 'strike', 'expiry', 'side'])['delta'].diff()
    
    # Calculate Δ-Delta percentage
    df['delta_change_perc'] = df.groupby(['underlying', 'strike', 'expiry', 'side'])['delta'].pct_change() * 100
    
    # Calculate Delta velocity (change per minute)
    # how many minutes passed since the last data point
    df['time_diff_minutes'] = df.groupby(['underlying', 'strike', 'expiry', 'side'])['timestamp_ist'].diff().dt.total_seconds() / 60

    # Calculate Delta velocity (change per minute)
    df['delta_velocity'] = df['delta_change'] / df['time_diff_minutes']
    
    # Add Delta momentum categories
    df['delta_momentum'] = 'STABLE'
    
    # Accelerating: Delta change is increasing
    accelerating_conditions = (
        (df['delta_change'].abs() > delta_change_threshold) &
        (df['delta_velocity'].abs() > delta_velocity_threshold) &
        (df['delta_change'] > 0)
    )
    df.loc[accelerating_conditions, 'delta_momentum'] = 'ACCELERATING'
    
    # Decelerating: Delta change is decreasing
    decelerating_conditions = (
        (df['delta_change'].abs() > delta_change_threshold) &
        (df['delta_velocity'].abs() > delta_velocity_threshold) &
        (df['delta_change'] < 0)
    )
    df.loc[decelerating_conditions, 'delta_momentum'] = 'DECELERATING'
    
    # Show Δ-Delta statistics
    print(f"\n📈 Δ-Delta Feature Statistics:")
    print(f"Delta change range: {df['delta_change'].min():.4f} to {df['delta_change'].max():.4f}")
    print(f"Delta change % range: {df['delta_change_perc'].min():.2f}% to {df['delta_change_perc'].max():.2f}%")
    print(f"Delta velocity range: {df['delta_velocity'].min():.6f} to {df['delta_velocity'].max():.6f}")
    
    delta_momentum_counts = df['delta_momentum'].value_counts()
    print(f"\n🎯 Delta Momentum Distribution:")
    for momentum, count in delta_momentum_counts.items():
        print(f"{momentum}: {count} records")
    
    # Clean up temporary columns
    df = df.drop(['time_diff_minutes'], axis=1)
    
    return df

def get_master_features( underlying, start_date, expiry=None, limit=None, latest_only=False, strikes=None):
    """
        Get (Option Chain + OI Trigger + OI Trap + Δ-Delta) features

        Args:
            underlying (str): Stock/index symbol (e.g., 'NIFTY', 'BANKNIFTY')
            start_date (str): Start date in 'YYYY-MM-DD' format
            expiry (str, optional): Expiry date as string 'YYYY-MM-DD'. If None, include all expiries.
            limit (int, optional): Max number of records to return. If None, return all records.
            latest_only (bool, optional): If True, return only the most recent timestamp. Default is False.
            strikes (list[int], optional): List of strikes to include. If None, include all strikes.
        
        Returns:
            pandas.DataFrame: Standardized features ready for ML
    """

    # Base WHERE clause
    where_clause = f"""
        oc.snap_minute_ist >= '{start_date}'
        AND oc.underlying = '{underlying}'
        AND oc.delta IS NOT NULL
        AND oc.iv IS NOT NULL
        AND oc.ltp > 0
    """

    # Add expiry filter only if provided
    if expiry is not None:
        where_clause += f" AND oc.expiry = '{expiry}'"

    # Add strike filter only if provided
    if strikes is not None and len(strikes) > 0:
        strike_list = ', '.join(map(str, strikes))
        where_clause += f" AND oc.strike IN ({strike_list})"

    # Build the base query
    query = f"""
    SELECT 
        -- Standardized timestamp (IST)
        oc.snap_minute_ist as timestamp_ist,

        -- Option Chain Features
        oc.underlying,
        oc.expiry,
        oc.strike,
        oc.side,
        oc.ltp,
        oc.bid_price,
        oc.ask_price,
        oc.volume,
        oc.oi,
        oc.prev_oi,

        -- Greeks (Delta, IV, etc.)
        oc.delta,
        oc.iv,
        oc.theta,
        oc.vega,
        oc.gamma,
        oc.rho,

        -- Calculated OI Change
        oc.oi - oc.prev_oi as oi_change,

        -- OI Change Percentage
        CASE 
            WHEN oc.prev_oi > 0 THEN ((oc.oi - oc.prev_oi) / oc.prev_oi) * 100
            ELSE NULL 
        END as oi_change_perc

    FROM market.option_chain_1m oc
    WHERE {where_clause}
    ORDER BY oc.snap_minute_ist ASC, oc.expiry, oc.strike, oc.side
    """

    # Add LIMIT only if provided
    if limit is not None:
        query += f"\nLIMIT {limit}"

    df = conn.query_df(query)

    # Add OI Trigger logic
    df = add_oi_trigger(df)

    # Add Trap logic
    df = add_oi_trap(df)

    # Add Δ-Delta feature
    df = add_delta_change(df)

    print(f"✅ Retrieved {len(df)} standardized records for {underlying}")
    if expiry:
        print(f"📅 Filtered for expiry: {expiry}")
    if strikes:
        print(f"🎯 Filtered for strikes: {strikes}")
    if limit:
        print(f"🔢 Limited to {limit} rows")
    print(f"📊 Added OI change calculations, oi_trigger, oi_trap, and Δ-Delta columns")

    # If latest_only is True, filter for the most recent timestamp
    if latest_only and not df.empty:
        latest_timestamp = df['timestamp_ist'].max()
        df = df[df['timestamp_ist'] == latest_timestamp]
        print(f"📅 Filtered for latest timestamp: {latest_timestamp}")

    return df

#! ------------ UI ------------

st.set_page_config(page_title="📊 Option Chain (Live)", layout="wide")
st.title("📊 Option Chain — Live")

#! ------------ Set (by default) -- universe:index | index_choice:NIFTY | stock_choice:RELIANCE ------------

ensure_session_defaults()

#! ------------ ClickHouse connection check ------------

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


#! ------------ Left Bar ------------

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

    #! Parameter -> 2 : Expiry
    today = datetime.date.today().isoformat()
    expiry_rows = conn.query(
        """
            SELECT DISTINCT expiry FROM instruments
            WHERE instrument_type = 'OPT' AND underlying = %(symbol)s AND expiry >= %(today)s
            ORDER BY expiry
        """, parameters={"symbol": SYMBOL, "today": today}
    ).result_rows
    expiry_choices = [str(row[0]) for row in expiry_rows]
    SELECTED_EXPIRY = st.selectbox("Expiry", expiry_choices, key="expiry") if expiry_choices else None

    #! Parameter -> 3 : Strikecount
    STRIKECOUNT = st.number_input("Strikecount (± per side, max 50)", min_value=1, max_value=10, value=1, step=1)

    #! Parameter -> 4 : Refresh interval
    REFRESH_MS = st.number_input("Refresh interval (ms)", min_value=5000, max_value=60000, value=5000, step=2000)

    #! Caption
    st.divider()
    st.caption("ATM is computed from underlying LTP of the same symbol via ClickHouse.")

#! ------------ Autorefresh ------------
st_autorefresh(interval=REFRESH_MS, key="_live_oc_refresh")


#! ------------ Session storage for small history of summary (optional) ------------
if "_snapshots" not in st.session_state:
    st.session_state["_snapshots"] = []


#! ====================== Live fetch ======================
today_date = datetime.datetime.today().strftime('%Y-%m-%d')

#! Get list of strikes based on StrikeCount and ATM Strike
ul_ltp = float(st.session_state["td_analytics"].get_spot_ltp(SYMBOL).LTP.iloc[0])
strike_step = get_strike_step(SYMBOL)
atm_strike = round(ul_ltp / strike_step) * strike_step
strikes = [atm_strike + (i * strike_step) for i in range(-STRIKECOUNT, STRIKECOUNT + 1)]
strikes = sorted(strikes)

#! Option Chain Table
df_oc = get_master_features(underlying=SYMBOL, start_date=today_date, limit=None, expiry=SELECTED_EXPIRY, latest_only=True, strikes=strikes)

#! Extract latest timestamp from option chain data
if not df_oc.empty and 'timestamp_ist' in df_oc.columns:
    latest_timestamp = df_oc['timestamp_ist'].max()
    # Format timestamp for display (e.g., "14:30:15" or "14:30")
    if pd.api.types.is_datetime64_any_dtype(df_oc['timestamp_ist']):
        as_on_time = latest_timestamp.strftime("%H:%M:%S")
    else:
        # If it's a string, try to parse it
        try:
            latest_timestamp = pd.to_datetime(latest_timestamp)
            as_on_time = latest_timestamp.strftime("%H:%M:%S")
        except:
            as_on_time = str(latest_timestamp)
else:
    as_on_time = "N/A"

#! Metrics Row Above Option Chain
col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a:
    st.metric("Underlying LTP", f"{ul_ltp:.2f}")
with col_b:
    st.metric("ATM Strike", f"{atm_strike}")
with col_c:
    st.metric("PCR (Total)", "")
with col_d:
    st.metric("India VIX", "")
with col_e:
    st.metric("As On", as_on_time)

#! Option Chain Table (Classic Layout)
ce_cols = ['iv', 'oi', 'prev_oi', 'oi_change', 'oi_change_perc','oi_trigger','oi_trap', 'volume', 'ltp', 'bid_price', 'ask_price']
pe_cols = ['bid_price', 'ask_price', 'ltp', 'volume', 'oi', 'prev_oi', 'oi_change', 'oi_change_perc','oi_trigger','oi_trap', 'iv']

ce_df = df_oc[df_oc['side'] == 'C'][['strike'] + ce_cols].set_index('strike')
pe_df = df_oc[df_oc['side'] == 'P'][['strike'] + pe_cols].set_index('strike')
ce_df.columns = pd.MultiIndex.from_product([['CE'], ce_df.columns])
pe_df.columns = pd.MultiIndex.from_product([['PE'], pe_df.columns])
ce_df = ce_df.reindex(strikes)
pe_df = pe_df.reindex(strikes)
strike_col = pd.DataFrame({'STRIKE': strikes}, index= strikes)
strike_col.columns = pd.MultiIndex.from_tuples([('', 'STRIKE')])
chain_df = pd.concat([ce_df, strike_col, pe_df], axis=1)
chain_df.reset_index(drop=True, inplace=True)

mid = len(ce_df.columns)
cols = list(ce_df.columns) + [('', 'STRIKE')] + list(pe_df.columns)
chain_df = chain_df[cols]
atm_highlight = lambda row: ['background-color: rgba(255, 215, 0, 0.20)' if row[('', 'STRIKE')] == atm_strike else '' for _ in row]

st.subheader("Option Chain (OI Trap & OI Trigger)")
st.dataframe(chain_df.style.apply(atm_highlight, axis=1))

#! Option Greeks Table (Classic Layout)
greeks_cols = ['delta', 'theta', 'gamma', 'vega', 'rho', 'ltp']
ce_g_df = df_oc[df_oc['side'] == 'C'][['strike'] + greeks_cols].set_index('strike')
pe_g_df = df_oc[df_oc['side'] == 'P'][['strike'] + greeks_cols].set_index('strike')
ce_g_df.columns = pd.MultiIndex.from_product([['CE'], ce_g_df.columns])
pe_g_df.columns = pd.MultiIndex.from_product([['PE'], pe_g_df.columns])
ce_g_df = ce_g_df.reindex(strikes)
pe_g_df = pe_g_df.reindex(strikes)

# Separate LTP columns from other Greeks
ce_greeks_only = ce_g_df[[col for col in ce_g_df.columns if col[1] != 'ltp']]
ce_ltp = ce_g_df[[('CE', 'ltp')]]
pe_greeks_only = pe_g_df[[col for col in pe_g_df.columns if col[1] != 'ltp']]
pe_ltp = pe_g_df[[('PE', 'ltp')]]

strike_col_g = pd.DataFrame({'STRIKE': strikes}, index=strikes)
strike_col_g.columns = pd.MultiIndex.from_tuples([('', 'STRIKE')])

# Concatenate in the desired order: CE Greeks -> CE LTP -> STRIKE -> PE LTP -> PE Greeks
greeks_df = pd.concat([ce_greeks_only, ce_ltp, strike_col_g, pe_ltp, pe_greeks_only], axis=1)
greeks_df.reset_index(drop=True, inplace=True)

# Define column order: CE Greeks -> CE LTP -> STRIKE -> PE LTP -> PE Greeks
gcols = list(ce_greeks_only.columns) + [('CE', 'ltp')] + [('', 'STRIKE')] + [('PE', 'ltp')] + list(pe_greeks_only.columns)
greeks_df = greeks_df[gcols]
st.subheader("Option Greeks")
st.dataframe(greeks_df.style.apply(atm_highlight, axis=1))

#! Delta Change Table
delta_change_cols = ['delta', 'delta_change', 'delta_change_perc', 'delta_velocity', 'delta_momentum','ltp']
ce_dc_df = df_oc[df_oc['side'] == 'C'][['strike'] + delta_change_cols].set_index('strike')
pe_dc_df = df_oc[df_oc['side'] == 'P'][['strike'] + delta_change_cols].set_index('strike')
ce_dc_df.columns = pd.MultiIndex.from_product([['CE'], ce_dc_df.columns])
pe_dc_df.columns = pd.MultiIndex.from_product([['PE'], pe_dc_df.columns])
ce_dc_df = ce_dc_df.reindex(strikes)
pe_dc_df = pe_dc_df.reindex(strikes)

# Separate LTP columns from other Delta Change columns
ce_delta_only = ce_dc_df[[col for col in ce_dc_df.columns if col[1] != 'ltp']]
ce_ltp_dc = ce_dc_df[[('CE', 'ltp')]]
pe_delta_only = pe_dc_df[[col for col in pe_dc_df.columns if col[1] != 'ltp']]
pe_ltp_dc = pe_dc_df[[('PE', 'ltp')]]

strike_col_dc = pd.DataFrame({'STRIKE': strikes}, index=strikes)
strike_col_dc.columns = pd.MultiIndex.from_tuples([('', 'STRIKE')])

# Concatenate in the desired order: CE Delta -> CE LTP -> STRIKE -> PE LTP -> PE Delta
delta_change_df = pd.concat([ce_delta_only, ce_ltp_dc, strike_col_dc, pe_ltp_dc, pe_delta_only], axis=1)
delta_change_df.reset_index(drop=True, inplace=True)

# Define column order: CE Delta -> CE LTP -> STRIKE -> PE LTP -> PE Delta
dcols = list(ce_delta_only.columns) + [('CE', 'ltp')] + [('', 'STRIKE')] + [('PE', 'ltp')] + list(pe_delta_only.columns)
delta_change_df = delta_change_df[dcols]
st.subheader("Delta Change")
st.dataframe(delta_change_df.style.apply(atm_highlight, axis=1))