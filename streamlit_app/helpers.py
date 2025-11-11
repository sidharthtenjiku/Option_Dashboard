import streamlit as st
from clickhouse_connect import get_client

#! Common Helping Functions

def get_clickhouse_conn(host, port, username, password, database):
    return get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database
    )
    
def ensure_session_defaults():
    ss = st.session_state
    ss.setdefault("universe", "index")       # "index" | "stocks"
    ss.setdefault("index_choice", "NIFTY")   # default index label
    ss.setdefault("stock_choice", "RELIANCE")# default stock label


def get_strike_step(symbol):
    if "BANKNIFTY" in symbol:
        return 100
    elif "NIFTY" in symbol:
        return 50
    elif "FINNIFTY" in symbol:
        return 50
    return 1