import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_autorefresh import st_autorefresh
import sqlite3
from datetime import datetime,timezone,timedelta,time as dtime
import threading
import logging
import pytz
import json
import os
import streamlit as st
from pathlib import Path
from clickhouse_connect import get_client
from truedata import TD_live
from truedata.analytics import TD_analytics
from config import path_holiday_json, host, port, username, password, database, td_username, td_password, td_log_level
from helpers import get_clickhouse_conn

#! ------------ Must be the first Streamlit command ------------
st.set_page_config(page_title="Options Dashboard", layout="wide")

st.title("🏠 Home")
st.write("Welcome to the Options Dashboard.")

TODAY = datetime.now().strftime("%Y%m%d")

#! Helpig Functions

IST = pytz.timezone("Asia/Kolkata")
IST = timezone(timedelta(hours=5, minutes=30))

def load_holidays_ist(path_holiday_json: str | Path | None) -> set[str]:
    if not path_holiday_json:
        return set()
    p = Path(path_holiday_json)
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            return set()
    return set()

def try_connect_clickhouse(host, port, username, password, database):
    try:
        conn = get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        conn.query("SELECT 1")
        st.session_state["clickhouse_conn"] = conn
        st.session_state["clickhouse_status"] = "connected"
    except Exception as e:
        st.session_state["clickhouse_conn"] = None
        st.session_state["clickhouse_status"] = f"error: {e}"


#! Show connect button if not connected, or show status
if "clickhouse_status" not in st.session_state or st.session_state["clickhouse_status"] != "connected":
    if st.button("Connect to ClickHouse", type="primary"):
        try_connect_clickhouse(host, port, username, password, database)

if "td_analytics" not in st.session_state:
    st.session_state["td_analytics"] = TD_analytics(td_username, td_password, td_log_level)

#! Show status indicator
status = st.session_state.get("clickhouse_status", "disconnected")
if status == "connected":
    st.success("✅ ClickHouse Connected")
elif status.startswith("error:"):
    st.error(f"❌ ClickHouse Connection Error: {status[6:]}")
else:
    st.info("ClickHouse not connected")

if "clickhouse_conn" not in st.session_state:
    st.session_state["clickhouse_conn"] = get_clickhouse_conn(host, port, username, password, database)


#! ---------- Futures/Options background collector ----------

HOLIDAYS_IST = load_holidays_ist(path_holiday_json)
