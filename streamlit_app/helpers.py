import streamlit as st

#! ------------ Streamlit Session Defualt ------------
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