from helpers import get_clickhouse_conn, ensure_session_defaults
import datetime
from nselib import derivatives
import plotly.express as px
import os
import pytz
import time
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from clickhouse_connect import get_client
from truedata.analytics import TD_analytics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import host, port, username, password, database, path_fiidii_data_files

import warnings
warnings.filterwarnings("ignore")
IST = pytz.timezone("Asia/Kolkata")

#! ================== PATHS (for FII DII data) ==================

PATH_FIIDII_DATA_FILES = os.path.join(path_fiidii_data_files, 'data')
PATH_RAW_FILE = os.path.join(PATH_FIIDII_DATA_FILES, '1_participants_raw_oi_nse.csv')
PATH_OUTPUT_FILE = os.path.join(PATH_FIIDII_DATA_FILES, '2_net_output_oi_data.csv')

#! ================== FUNCTIONS ==================

def fetch_and_save_open_interest_data(date,PATH_RAW_FILE):
    try:
        data = derivatives.participant_wise_open_interest(trade_date=date)

        if isinstance(data, pd.DataFrame) and data.empty:
            print(f"No data available for {date}. Skipping...")
            return False

        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data

        df['Date'] = date
        cols = ['Date'] + [col for col in df.columns if col != 'Date']
        df = df[cols]

        if os.path.exists(PATH_RAW_FILE):
            df.to_csv(PATH_RAW_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(PATH_RAW_FILE, mode='w', header=True, index=False)

        print(f"Data for {date} saved successfully.")
        return True

    except Exception as e:
        print(f"Error fetching data for {date}: {e}")
        return False

def date_range(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += datetime.timedelta(days=1)

def get_last_saved_date(PATH_RAW_FILE):
    if os.path.exists(PATH_RAW_FILE):
        df = pd.read_csv(PATH_RAW_FILE)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        last_date = df['Date'].max()
        return last_date
    return None 

def safe_float(val):
    '''
        Helper function to safely convert to float (used across all visualizations)
        Args:
            val: Value to convert to float
        Returns:
            float: Converted value
    '''
    try:
        result = pd.to_numeric(val, errors='coerce')
        return float(result) if pd.notna(result) else 0.0
    except:
        return 0.0

#! Main

def fetch_and_save_raw_historical_data(START_DATE_STR,PATH_RAW_FILE):
    '''
        Fetch historical Open Interest (CLIENT|FII|DII|PRO|TOTAL) data from NSE and save it as raw data to the CSV file (1_participants_raw_oi_nse.csv)
        Args:
            START_DATE_STR: Start date in the format 'dd-mm-yyyy'
        Returns:
            bool: True if at least one date was successfully fetched, False otherwise
    '''
    start_date = datetime.datetime.strptime(START_DATE_STR, '%d-%m-%Y')
    end_date = datetime.datetime.now()
    last_saved_date = get_last_saved_date(PATH_RAW_FILE)
    if last_saved_date:
        print(f"Last saved date: {last_saved_date.strftime('%d-%m-%Y')}")
        start_date = last_saved_date + datetime.timedelta(days=1)

    at_least_one_success = False
    for current_date in date_range(start_date, end_date):
        current_date_str = current_date.strftime('%d-%m-%Y')
        success = fetch_and_save_open_interest_data(current_date_str,PATH_RAW_FILE)
        if success:
            at_least_one_success = True
            time.sleep(2)  
        else:
            print(f"Skipping {current_date_str} due to no data or error.")
    
    return at_least_one_success

def create_and_save_net_output_data(PATH_RAW_FILE,PATH_OUTPUT_FILE):

    #! Read the raw oi participants data file | Convert the 'Date' column to datetime format | Sort the data by 'Date' and 'CLIENT_TYPE'
    df = pd.read_csv(PATH_RAW_FILE)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df_sorted = df.sort_values(by=['Date', 'CLIENT_TYPE'])

    #! Initialize the empty output dataframe with the required columns
    output_df = pd.DataFrame(columns=[
        'Date', 
        'CLIENT_TYPE',
        'Future Index Long', 
        'Future Index Short', 
        'Future Stock Long', 
        'Future Stock Short',
        'Option Index Call Long',
        'Option Index Put Long',
        'Option Index Call Short',
        'Option Index Put Short',
        'Option Stock Call Long',
        'Option Stock Put Long',
        'Option Stock Call Short',
        'Option Stock Put Short',
        'Net Buy/Sell Index Futures',
        'Net Buy/Sell Stock Futures',
        'Net Buy/Sell Index Call',
        'Net Buy/Sell Index Put',
        'Net Buy/Sell Stock Call',
        'Net Buy/Sell Stock Put',
        'Total Positions carried Today Index Futures',
        'Total Positions carried 1 Day Ago Index Futures',
        'Total Positions carried 2 Days Ago Index Futures',
        'Total Positions carried 3 Days Ago Index Futures',
        'Total Positions carried 4 Days Ago Index Futures',
        'Total Positions carried 5 Days Ago Index Futures',
        'Total Positions carried 6 Days Ago Index Futures',
        'Total Positions carried 7 Days Ago Index Futures',
        'Total Positions carried Today Stock Futures',
        'Total Positions carried 1 Day Ago Stock Futures',
        'Total Positions carried 2 Days Ago Stock Futures',
        'Total Positions carried 3 Days Ago Stock Futures',
        'Total Positions carried 4 Days Ago Stock Futures',
        'Total Positions carried 5 Days Ago Stock Futures',
        'Total Positions carried 6 Days Ago Stock Futures',
        'Total Positions carried 7 Days Ago Stock Futures',
        'Total Positions carried Today Index Call',
        'Total Positions carried 1 Day Ago Index Call',
        'Total Positions carried 2 Days Ago Index Call',
        'Total Positions carried 3 Days Ago Index Call',
        'Total Positions carried 4 Days Ago Index Call',
        'Total Positions carried 5 Days Ago Index Call',
        'Total Positions carried 6 Days Ago Index Call',
        'Total Positions carried 7 Days Ago Index Call',
        'Total Positions carried Today Index Put',
        'Total Positions carried 1 Day Ago Index Put',
        'Total Positions carried 2 Days Ago Index Put',
        'Total Positions carried 3 Days Ago Index Put',
        'Total Positions carried 4 Days Ago Index Put',
        'Total Positions carried 5 Days Ago Index Put',
        'Total Positions carried 6 Days Ago Index Put',
        'Total Positions carried 7 Days Ago Index Put',
        'Total Positions carried Today Stock Call',
        'Total Positions carried 1 Day Ago Stock Call',
        'Total Positions carried 2 Days Ago Stock Call',
        'Total Positions carried 3 Days Ago Stock Call',
        'Total Positions carried 4 Days Ago Stock Call',
        'Total Positions carried 5 Days Ago Stock Call',
        'Total Positions carried 6 Days Ago Stock Call',
        'Total Positions carried 7 Days Ago Stock Call',
        'Total Positions carried Today Stock Put',
        'Total Positions carried 1 Day Ago Stock Put',
        'Total Positions carried 2 Days Ago Stock Put',
        'Total Positions carried 3 Days Ago Stock Put',
        'Total Positions carried 4 Days Ago Stock Put',
        'Total Positions carried 5 Days Ago Stock Put',
        'Total Positions carried 6 Days Ago Stock Put',
        'Total Positions carried 7 Days Ago Stock Put',
    ])

    #! Loop through each unique client type
    for client_type in df_sorted['CLIENT_TYPE'].unique():

        if client_type == 'Total':
            continue

        client_data = df_sorted[df_sorted['CLIENT_TYPE'] == client_type]
        
        temp_changes = []

        for date in client_data['Date'].unique():
        
            today_row = client_data[client_data['Date'] == date]

            if date == client_data['Date'].min():
            
                changes_row = {
                    'Date': date,
                    'CLIENT_TYPE': client_type,
                    'Future Index Long': None,
                    'Future Index Short': None,
                    'Future Stock Long': None,
                    'Future Stock Short': None,
                    'Option Index Call Long': None,
                    'Option Index Put Long': None,
                    'Option Index Call Short': None,
                    'Option Index Put Short': None,
                    'Option Stock Call Long': None,
                    'Option Stock Put Long': None,
                    'Option Stock Call Short': None,
                    'Option Stock Put Short': None,
                    'Net Buy/Sell Index Futures': None,
                    'Net Buy/Sell Stock Futures': None,
                    'Net Buy/Sell Index Call': None,
                    'Net Buy/Sell Index Put': None,
                    'Net Buy/Sell Stock Call': None,
                    'Net Buy/Sell Stock Put': None,
                    'Total Positions carried Today Index Futures': None,
                    'Total Positions carried 1 Day Ago Index Futures': None,
                    'Total Positions carried 2 Days Ago Index Futures': None,
                    'Total Positions carried 3 Days Ago Index Futures': None,
                    'Total Positions carried 4 Days Ago Index Futures': None,
                    'Total Positions carried 5 Days Ago Index Futures': None,
                    'Total Positions carried 6 Days Ago Index Futures': None,
                    'Total Positions carried 7 Days Ago Index Futures': None,
                    'Total Positions carried Today Stock Futures': None,
                    'Total Positions carried 1 Day Ago Stock Futures': None,
                    'Total Positions carried 2 Days Ago Stock Futures': None,
                    'Total Positions carried 3 Days Ago Stock Futures': None,
                    'Total Positions carried 4 Days Ago Stock Futures': None,
                    'Total Positions carried 5 Days Ago Stock Futures': None,
                    'Total Positions carried 6 Days Ago Stock Futures': None,
                    'Total Positions carried 7 Days Ago Stock Futures': None,
                    'Total Positions carried Today Index Call': None,
                    'Total Positions carried 1 Day Ago Index Call': None,
                    'Total Positions carried 2 Days Ago Index Call': None,
                    'Total Positions carried 3 Days Ago Index Call': None,
                    'Total Positions carried 4 Days Ago Index Call': None,
                    'Total Positions carried 5 Days Ago Index Call': None,
                    'Total Positions carried 6 Days Ago Index Call': None,
                    'Total Positions carried 7 Days Ago Index Call': None,
                    'Total Positions carried Today Index Put': None,
                    'Total Positions carried 1 Day Ago Index Put': None,
                    'Total Positions carried 2 Days Ago Index Put': None,
                    'Total Positions carried 3 Days Ago Index Put': None,
                    'Total Positions carried 4 Days Ago Index Put': None,
                    'Total Positions carried 5 Days Ago Index Put': None,
                    'Total Positions carried 6 Days Ago Index Put': None,
                    'Total Positions carried 7 Days Ago Index Put': None,
                    'Total Positions carried Today Stock Call': None,
                    'Total Positions carried 1 Day Ago Stock Call': None,
                    'Total Positions carried 2 Days Ago Stock Call': None,
                    'Total Positions carried 3 Days Ago Stock Call': None,
                    'Total Positions carried 4 Days Ago Stock Call': None,
                    'Total Positions carried 5 Days Ago Stock Call': None,
                    'Total Positions carried 6 Days Ago Stock Call': None,
                    'Total Positions carried 7 Days Ago Stock Call': None,
                    'Total Positions carried Today Stock Put': None,
                    'Total Positions carried 1 Day Ago Stock Put': None,
                    'Total Positions carried 2 Days Ago Stock Put': None,
                    'Total Positions carried 3 Days Ago Stock Put': None,
                    'Total Positions carried 4 Days Ago Stock Put': None,
                    'Total Positions carried 5 Days Ago Stock Put': None,
                    'Total Positions carried 6 Days Ago Stock Put': None,
                    'Total Positions carried 7 Days Ago Stock Put': None,
                }
            else:
                prev_dates = client_data[client_data['Date'] < date]['Date'].unique()
                # st.write(prev_dates)
                if len(prev_dates) >= 7:
                    prev_date_1 = prev_dates[-1]
                    prev_date_2 = prev_dates[-2]
                    prev_date_3 = prev_dates[-3]
                    prev_date_4 = prev_dates[-4]
                    prev_date_5 = prev_dates[-5]
                    prev_date_6 = prev_dates[-6]
                    prev_date_7 = prev_dates[-7]
                else:
                    
                    prev_date_1 = prev_dates[0] if len(prev_dates) > 0 else client_data['Date'].min()
                    prev_date_2 = prev_date_1
                    prev_date_3 = prev_date_2
                    prev_date_4 = prev_date_3
                    prev_date_5 = prev_date_4
                    prev_date_6 = prev_date_5
                    prev_date_7 = prev_date_6

                prev_row_1 = client_data[client_data['Date'] == prev_date_1]
                prev_row_2 = client_data[client_data['Date'] == prev_date_2]
                prev_row_3 = client_data[client_data['Date'] == prev_date_3]
                prev_row_4 = client_data[client_data['Date'] == prev_date_4]
                prev_row_5 = client_data[client_data['Date'] == prev_date_5]
                prev_row_6 = client_data[client_data['Date'] == prev_date_6]
                prev_row_7 = client_data[client_data['Date'] == prev_date_7]
            
                changes_row = {
                    'Date': date,
                    'CLIENT_TYPE': client_type,
                    'Future Index Long': today_row['Future Index Long'].values[0] - prev_row_1['Future Index Long'].values[0],
                    'Future Index Short': today_row['Future Index Short'].values[0] - prev_row_1['Future Index Short'].values[0],
                    'Future Stock Long': today_row['Future Stock Long'].values[0] - prev_row_1['Future Stock Long'].values[0],
                    'Future Stock Short': today_row['Future Stock Short'].values[0] - prev_row_1['Future Stock Short'].values[0],
                    'Option Index Call Long': today_row['Option Index Call Long'].values[0] - prev_row_1['Option Index Call Long'].values[0],
                    'Option Index Put Long': today_row['Option Index Put Long'].values[0] - prev_row_1['Option Index Put Long'].values[0],
                    'Option Index Call Short': today_row['Option Index Call Short'].values[0] - prev_row_1['Option Index Call Short'].values[0],
                    'Option Index Put Short': today_row['Option Index Put Short'].values[0] - prev_row_1['Option Index Put Short'].values[0],
                    'Option Stock Call Long': today_row['Option Stock Call Long'].values[0] - prev_row_1['Option Stock Call Long'].values[0],
                    'Option Stock Put Long': today_row['Option Stock Put Long'].values[0] - prev_row_1['Option Stock Put Long'].values[0],
                    'Option Stock Call Short': today_row['Option Stock Call Short'].values[0] - prev_row_1['Option Stock Call Short'].values[0],
                    'Option Stock Put Short': today_row['Option Stock Put Short'].values[0] - prev_row_1['Option Stock Put Short'].values[0],
                    
                    'Net Buy/Sell Index Futures': (today_row['Future Index Long'].values[0] - today_row['Future Index Short'].values[0]) - 
                                                (prev_row_1['Future Index Long'].values[0] - prev_row_1['Future Index Short'].values[0]),
                    'Net Buy/Sell Stock Futures': (today_row['Future Stock Long'].values[0] - today_row['Future Stock Short'].values[0]) - 
                                                (prev_row_1['Future Stock Long'].values[0] - prev_row_1['Future Stock Short'].values[0]),
                    'Net Buy/Sell Index Call': (today_row['Option Index Call Long'].values[0] - today_row['Option Index Call Short'].values[0]) - 
                                            (prev_row_1['Option Index Call Long'].values[0] - prev_row_1['Option Index Call Short'].values[0]),
                    'Net Buy/Sell Index Put': (today_row['Option Index Put Long'].values[0] - today_row['Option Index Put Short'].values[0]) - 
                                            (prev_row_1['Option Index Put Long'].values[0] - prev_row_1['Option Index Put Short'].values[0]),
                    'Net Buy/Sell Stock Call': (today_row['Option Stock Call Long'].values[0] - today_row['Option Stock Call Short'].values[0]) - 
                                            (prev_row_1['Option Stock Call Long'].values[0] - prev_row_1['Option Stock Call Short'].values[0]),
                    'Net Buy/Sell Stock Put': (today_row['Option Stock Put Long'].values[0] - today_row['Option Stock Put Short'].values[0]) - 
                                            (prev_row_1['Option Stock Put Long'].values[0] - prev_row_1['Option Stock Put Short'].values[0]),
            
                    'Total Positions carried Today Index Futures': today_row['Future Index Long'].values[0] - today_row['Future Index Short'].values[0],
                    'Total Positions carried 1 Day Ago Index Futures': prev_row_1['Future Index Long'].values[0] - prev_row_1['Future Index Short'].values[0],
                    'Total Positions carried 2 Days Ago Index Futures': prev_row_2['Future Index Long'].values[0] - prev_row_2['Future Index Short'].values[0],
                    'Total Positions carried 3 Days Ago Index Futures': prev_row_3['Future Index Long'].values[0] - prev_row_3['Future Index Short'].values[0],
                    'Total Positions carried 4 Days Ago Index Futures': prev_row_4['Future Index Long'].values[0] - prev_row_4['Future Index Short'].values[0],
                    'Total Positions carried 5 Days Ago Index Futures': prev_row_5['Future Index Long'].values[0] - prev_row_5['Future Index Short'].values[0],
                    'Total Positions carried 6 Days Ago Index Futures': prev_row_6['Future Index Long'].values[0] - prev_row_6['Future Index Short'].values[0],
                    'Total Positions carried 7 Days Ago Index Futures': prev_row_7['Future Index Long'].values[0] - prev_row_7['Future Index Short'].values[0],
                    'Total Positions carried Today Stock Futures': today_row['Future Stock Long'].values[0] - today_row['Future Stock Short'].values[0],
                    'Total Positions carried 1 Day Ago Stock Futures': prev_row_1['Future Stock Long'].values[0] - prev_row_1['Future Stock Short'].values[0],
                    'Total Positions carried 2 Days Ago Stock Futures': prev_row_2['Future Stock Long'].values[0] - prev_row_2['Future Stock Short'].values[0],
                    'Total Positions carried 3 Days Ago Stock Futures': prev_row_3['Future Stock Long'].values[0] - prev_row_3['Future Stock Short'].values[0],
                    'Total Positions carried 4 Days Ago Stock Futures': prev_row_4['Future Stock Long'].values[0] - prev_row_4['Future Stock Short'].values[0],
                    'Total Positions carried 5 Days Ago Stock Futures': prev_row_5['Future Stock Long'].values[0] - prev_row_5['Future Stock Short'].values[0],
                    'Total Positions carried 6 Days Ago Stock Futures': prev_row_6['Future Stock Long'].values[0] - prev_row_6['Future Stock Short'].values[0],
                    'Total Positions carried 7 Days Ago Stock Futures': prev_row_7['Future Stock Long'].values[0] - prev_row_7['Future Stock Short'].values[0],
                    'Total Positions carried Today Index Call': today_row['Option Index Call Long'].values[0] - today_row['Option Index Call Short'].values[0],
                    'Total Positions carried 1 Day Ago Index Call': prev_row_1['Option Index Call Long'].values[0] - prev_row_1['Option Index Call Short'].values[0],
                    'Total Positions carried 2 Days Ago Index Call': prev_row_2['Option Index Call Long'].values[0] - prev_row_2['Option Index Call Short'].values[0],
                    'Total Positions carried 3 Days Ago Index Call': prev_row_3['Option Index Call Long'].values[0] - prev_row_3['Option Index Call Short'].values[0],
                    'Total Positions carried 4 Days Ago Index Call': prev_row_4['Option Index Call Long'].values[0] - prev_row_4['Option Index Call Short'].values[0],
                    'Total Positions carried 5 Days Ago Index Call': prev_row_5['Option Index Call Long'].values[0] - prev_row_5['Option Index Call Short'].values[0],
                    'Total Positions carried 6 Days Ago Index Call': prev_row_6['Option Index Call Long'].values[0] - prev_row_6['Option Index Call Short'].values[0],
                    'Total Positions carried 7 Days Ago Index Call': prev_row_7['Option Index Call Long'].values[0] - prev_row_7['Option Index Call Short'].values[0],
                    'Total Positions carried Today Index Put': today_row['Option Index Put Long'].values[0] - today_row['Option Index Put Short'].values[0],
                    'Total Positions carried 1 Day Ago Index Put': prev_row_1['Option Index Put Long'].values[0] - prev_row_1['Option Index Put Short'].values[0],
                    'Total Positions carried 2 Days Ago Index Put': prev_row_2['Option Index Put Long'].values[0] - prev_row_2['Option Index Put Short'].values[0],
                    'Total Positions carried 3 Days Ago Index Put': prev_row_3['Option Index Put Long'].values[0] - prev_row_3['Option Index Put Short'].values[0],
                    'Total Positions carried 4 Days Ago Index Put': prev_row_4['Option Index Put Long'].values[0] - prev_row_4['Option Index Put Short'].values[0],
                    'Total Positions carried 5 Days Ago Index Put': prev_row_5['Option Index Put Long'].values[0] - prev_row_5['Option Index Put Short'].values[0],
                    'Total Positions carried 6 Days Ago Index Put': prev_row_6['Option Index Put Long'].values[0] - prev_row_6['Option Index Put Short'].values[0],
                    'Total Positions carried 7 Days Ago Index Put': prev_row_7['Option Index Put Long'].values[0] - prev_row_7['Option Index Put Short'].values[0],
                    'Total Positions carried Today Stock Call': today_row['Option Stock Call Long'].values[0] - today_row['Option Stock Call Short'].values[0],
                    'Total Positions carried 1 Day Ago Stock Call': prev_row_1['Option Stock Call Long'].values[0] - prev_row_1['Option Stock Call Short'].values[0],
                    'Total Positions carried 2 Days Ago Stock Call': prev_row_2['Option Stock Call Long'].values[0] - prev_row_2['Option Stock Call Short'].values[0],
                    'Total Positions carried 3 Days Ago Stock Call': prev_row_3['Option Stock Call Long'].values[0] - prev_row_3['Option Stock Call Short'].values[0],
                    'Total Positions carried 4 Days Ago Stock Call': prev_row_4['Option Stock Call Long'].values[0] - prev_row_4['Option Stock Call Short'].values[0],
                    'Total Positions carried 5 Days Ago Stock Call': prev_row_5['Option Stock Call Long'].values[0] - prev_row_5['Option Stock Call Short'].values[0],
                    'Total Positions carried 6 Days Ago Stock Call': prev_row_6['Option Stock Call Long'].values[0] - prev_row_6['Option Stock Call Short'].values[0],
                    'Total Positions carried 7 Days Ago Stock Call': prev_row_7['Option Stock Call Long'].values[0] - prev_row_7['Option Stock Call Short'].values[0],
                    'Total Positions carried Today Stock Put': today_row['Option Stock Put Long'].values[0] - today_row['Option Stock Put Short'].values[0],
                    'Total Positions carried 1 Day Ago Stock Put': prev_row_1['Option Stock Put Long'].values[0] - prev_row_1['Option Stock Put Short'].values[0],
                    'Total Positions carried 2 Days Ago Stock Put': prev_row_2['Option Stock Put Long'].values[0] - prev_row_2['Option Stock Put Short'].values[0],
                    'Total Positions carried 3 Days Ago Stock Put': prev_row_3['Option Stock Put Long'].values[0] - prev_row_3['Option Stock Put Short'].values[0],
                    'Total Positions carried 4 Days Ago Stock Put': prev_row_4['Option Stock Put Long'].values[0] - prev_row_4['Option Stock Put Short'].values[0],
                    'Total Positions carried 5 Days Ago Stock Put': prev_row_5['Option Stock Put Long'].values[0] - prev_row_5['Option Stock Put Short'].values[0],
                    'Total Positions carried 6 Days Ago Stock Put': prev_row_6['Option Stock Put Long'].values[0] - prev_row_6['Option Stock Put Short'].values[0],
                    'Total Positions carried 7 Days Ago Stock Put': prev_row_7['Option Stock Put Long'].values[0] - prev_row_7['Option Stock Put Short'].values[0],
                }

            temp_changes.append(changes_row)
        
        client_output_df = pd.DataFrame(temp_changes)
        output_df = pd.concat([output_df, client_output_df], ignore_index=True)

    #! Save the output dataframe to a CSV file
    output_df = output_df[output_df['CLIENT_TYPE'] != 'TOTAL']
    output_df.to_csv(PATH_OUTPUT_FILE, index=False)

    return output_df

#! ================== UI ==================

st.set_page_config(page_title="💰FII DII Report", layout="wide")
st.title("💰FII DII Report")

#! ================== Set (by default) -- universe:index | index_choice:NIFTY | stock_choice:RELIANCE ==================

ensure_session_defaults()

#! ================== ClickHouse connection check ==================

if "clickhouse_conn" not in st.session_state or st.session_state.get("clickhouse_status") != "connected":
    st.warning("Go to Home and connect to ClickHouse first.")
    st.stop()


conn = get_clickhouse_conn(host, port, username, password, database)

#! ================== Sidebar ==================
with st.sidebar:
    st.header("⚙️ Settings")

    #! Parameter 1: Universe / Date

    min_allowed_date = datetime.date(2012, 2, 1)
    max_allowed_date = datetime.date.today()

    selected_date = st.date_input(
        "Select a Date",
        min_value=min_allowed_date,
        max_value=max_allowed_date,
        value=max_allowed_date - datetime.timedelta(days=1)
    )

    #! Parameter 2: Days Ago Selection
    days_ago_options = ["None", "1 day ago", "2 days ago", "3 days ago", "4 days ago", "5 days ago", "6 days ago", "7 days ago"]
    selected_days_ago = st.radio(
        "Select Days Ago",
        options=days_ago_options,
        index=1,
        key="days_ago_radio"
    )

    #! Parameter 3 : Submit button
    submit_button = st.button("Submit")

#! ================== Submit button to show the FII|DII report for the selected date==================
if submit_button:

    #! ================== Getting the data table ==================

    output_df = pd.read_csv(PATH_OUTPUT_FILE, parse_dates=['Date'])
    output_df['Date'] = output_df['Date'].dt.date
    
    # Ensure selected date exists in the data
    if selected_date not in output_df['Date'].values:
        #! Fetch historical data and append to the raw data file (1_participants_raw_oi_nse.csv)
        fetch_success = fetch_and_save_raw_historical_data(selected_date.strftime('%d-%m-%Y'),PATH_RAW_FILE) 

        #! Process the data and save to the output file (2_net_output_oi_data.csv) only if fetch was successful
        if fetch_success:
            output_df = create_and_save_net_output_data(PATH_RAW_FILE,PATH_OUTPUT_FILE)
            output_df['Date'] = pd.to_datetime(output_df['Date']).dt.date
        else:
            st.warning("Failed to fetch historical data. Please try again later or try another previous date.")
    
    # Filter data by selected date
    filtered_data = output_df[output_df['Date'] == selected_date].copy()
    
    # Filter columns based on selected "days ago" value
    if selected_days_ago != "None":
        # Extract the number from "X day ago" or "X days ago"
        days_ago_num = selected_days_ago.split()[0]  # Gets "1", "2", etc.
        
        # Get all columns
        all_columns = filtered_data.columns.tolist()
        
        # Columns to keep (base columns + columns matching selected days ago)
        columns_to_keep = []
        
        # Always keep these base columns
        base_columns = ['Date', 'CLIENT_TYPE']
        columns_to_keep.extend(base_columns)
        
        # Keep all columns that are NOT "Total Positions carried X Days Ago" columns
        # EXCEPT for the selected days ago value
        for col in all_columns:
            if col in base_columns:
                continue
            
            # Check if this is a "Total Positions carried" column
            if 'Total Positions carried' in col:
                # Check if it matches the selected days ago value
                if f'{days_ago_num} Day Ago' in col or f'{days_ago_num} Days Ago' in col:
                    columns_to_keep.append(col)
                # Skip all other "Total Positions carried" columns (including "Today")
            else:
                # Keep all other columns (Net Buy/Sell, Future Index Long, etc.)
                columns_to_keep.append(col)
        
        # Filter the dataframe to show only selected columns
        filtered_data = filtered_data[columns_to_keep]
    
    # Display the filtered data
    st.subheader(f"{selected_date}")
    if selected_days_ago != "None":
        st.info(f"Showing columns for: {selected_days_ago}")
    st.dataframe(filtered_data)

    #! ================== Visualize the data ==================
    if not filtered_data.empty:

        #! 1. Participant Summary Dashboard (Bar chart)
        st.subheader("1. Net Futures/Options Exposure by Participant Type")
        if all(col in filtered_data.columns for col in [
            'CLIENT_TYPE',
            'Net Buy/Sell Index Futures', 'Net Buy/Sell Stock Futures',
            'Net Buy/Sell Index Call', 'Net Buy/Sell Index Put',
            'Net Buy/Sell Stock Call', 'Net Buy/Sell Stock Put'
        ]):
            fig = px.bar(
                filtered_data,
                x='CLIENT_TYPE',
                y=[
                    'Net Buy/Sell Index Futures',
                    'Net Buy/Sell Stock Futures',
                    'Net Buy/Sell Index Call',
                    'Net Buy/Sell Index Put',
                    'Net Buy/Sell Stock Call',
                    'Net Buy/Sell Stock Put'
                ],
                title='Both Futures/Options Summary',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            fig.update_layout(
                xaxis_title="Client Type",
                yaxis_title="Net Buy/Sell (Contracts)",
                legend_title="Segment",
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)

        # ========= Futures Summary =========
        futures_cols = [
            'Net Buy/Sell Index Futures',
            'Net Buy/Sell Stock Futures'
        ]

        fig_futures = px.bar(
            filtered_data,
            x='CLIENT_TYPE',
            y=futures_cols,
            barmode='group',
            title='Futures Summary',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # ========= Options Summary =========
        options_cols = [
            'Net Buy/Sell Index Call', 'Net Buy/Sell Index Put',
            'Net Buy/Sell Stock Call', 'Net Buy/Sell Stock Put'
        ]

        fig_options = px.bar(
            filtered_data,
            x='CLIENT_TYPE',
            y=options_cols,
            barmode='group',
            title='Options Summary',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_futures, use_container_width=True)
        with col2:
            st.plotly_chart(fig_options, use_container_width=True)

        #! 2. Futures Position Heatmap
        st.subheader("2. Futures Position Heatmap")
        if all(col in filtered_data.columns for col in ['CLIENT_TYPE', 'Future Index Long', 'Future Index Short', 'Future Stock Long', 'Future Stock Short']):
            heatmap_data = filtered_data[['CLIENT_TYPE', 'Future Index Long', 'Future Index Short', 'Future Stock Long', 'Future Stock Short']]
            heatmap_data = heatmap_data.melt(id_vars=['CLIENT_TYPE'], var_name='Segment', value_name='Contracts')

            fig = px.density_heatmap(
                heatmap_data,
                x='Segment',
                y='CLIENT_TYPE',
                z='Contracts',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

        #! 3. Options Call/Put Matrix (Bullish vs Bearish sentiment)
        st.subheader("3. Option Position Distribution by Participant")
        if all(col in filtered_data.columns for col in ['CLIENT_TYPE', 'Option Index Call Long', 'Option Index Call Short', 'Option Index Put Long', 'Option Index Put Short']):
            options_cols = [
                'Option Index Call Long', 'Option Index Call Short',
                'Option Index Put Long', 'Option Index Put Short'
            ]

            options_df = filtered_data.melt(id_vars=['CLIENT_TYPE'], value_vars=options_cols,
                                                var_name='Option Type', value_name='Contracts')

            fig = px.bar(
                options_df,
                x='CLIENT_TYPE',
                y='Contracts',
                color='Option Type',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        #! 4. Net Buy/Sell Gauge
        st.subheader("4. Total Net Buy/Sell Participant wise")
        net_columns = [
            'Net Buy/Sell Index Futures',
            'Net Buy/Sell Stock Futures',
            'Net Buy/Sell Index Call',
            'Net Buy/Sell Index Put',
            'Net Buy/Sell Stock Call',
            'Net Buy/Sell Stock Put'
        ]

        if all(col in filtered_data.columns for col in net_columns):
            # Create a copy and calculate total net value
            filtered_data_copy = filtered_data.copy()
            filtered_data_copy['Total_Net_Buy_Sell'] = filtered_data_copy[net_columns].sum(axis=1)

            # Define overall range for gauge scaling
            gauge_max = filtered_data_copy['Total_Net_Buy_Sell'].abs().max()

            # Create columns dynamically based on number of client types
            cols = st.columns(len(filtered_data_copy))

            # Display gauges side by side
            for i, (_, row) in enumerate(filtered_data_copy.iterrows()):
                with cols[i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=row['Total_Net_Buy_Sell'],
                        delta={
                            'reference': 0,
                            'increasing': {'color': 'green'},
                            'decreasing': {'color': 'red'}
                        },
                        title={'text': f"{row['CLIENT_TYPE']}"},
                        gauge={
                            'axis': {'range': [-gauge_max, gauge_max]},
                            'bar': {'color': 'green' if row['Total_Net_Buy_Sell'] > 0 else 'red'},
                            'steps': [
                                {'range': [-gauge_max, 0], 'color': 'rgba(255,0,0,0.1)'},
                                {'range': [0, gauge_max], 'color': 'rgba(0,255,0,0.1)'}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

        #! 5. Futures Long/Short Summary Tables
        st.subheader("5. Futures Long/Short Summary by Participant Type")
        if all(col in filtered_data.columns for col in ['CLIENT_TYPE', 'Future Index Long', 'Future Index Short', 'Future Stock Long', 'Future Stock Short']):
            # Define client types in order
            client_types = ['Client', 'FII', 'DII', 'Pro']
            
            # Create 4 columns for side-by-side display
            table_cols = st.columns(4)
            
            for idx, client_type in enumerate(client_types):
                # Filter data for this client type
                client_data = filtered_data[filtered_data['CLIENT_TYPE'] == client_type]
                
                if not client_data.empty:
                    row = client_data.iloc[0]
                    
                    # Calculate values for Row 1 (handle NaN/None values)
                    future_index_long = safe_float(row.get('Future Index Long', 0))
                    future_stock_long = safe_float(row.get('Future Stock Long', 0))
                    future_index_short = safe_float(row.get('Future Index Short', 0))
                    future_stock_short = safe_float(row.get('Future Stock Short', 0))
                    
                    long_value = future_index_long + future_stock_long
                    short_value = future_index_short + future_stock_short
                    total_value = long_value + short_value
                    
                    # Calculate percentages for Row 2
                    if total_value != 0:
                        long_percentage = (long_value / total_value) * 100
                        short_percentage = (short_value / total_value) * 100
                    else:
                        long_percentage = 0
                        short_percentage = 0
                    
                    # Create table data
                    table_data = {
                        '': ['Total Contracts', 'Percentage'],
                        'Long': [f'{long_value:,.0f}', f'{long_percentage:.2f}%'],
                        'Short': [f'{short_value:,.0f}', f'{short_percentage:.2f}%'],
                        'Total': [f'{total_value:,.0f}', '100.00%']
                    }
                    
                    table_df = pd.DataFrame(table_data)
                    
                    # Display table in the corresponding column
                    with table_cols[idx]:
                        st.markdown(f"**{client_type}**")
                        st.dataframe(
                            table_df.set_index(''),
                            use_container_width=True,
                            hide_index=False
                        )
                else:
                    # If no data for this client type, show empty table
                    with table_cols[idx]:
                        st.markdown(f"**{client_type}**")
                        empty_table = pd.DataFrame({
                            'Long': ['-', '-'],
                            'Short': ['-', '-'],
                            'Total': ['-', '-']
                        }, index=['Total Contracts', 'Percentage'])
                        st.dataframe(empty_table, use_container_width=True)

        #! 6. Today vs X Days Ago Comparison (only when days ago is selected)
        if selected_days_ago != "None" and not filtered_data.empty:
            days_ago_num = selected_days_ago.split()[0]
            st.subheader(f"6. Today vs {selected_days_ago} Comparison")
            
            # Load full data for comparison (we need both Today and X Days Ago columns)
            output_df_full = pd.read_csv(PATH_OUTPUT_FILE, parse_dates=['Date'])
            output_df_full['Date'] = output_df_full['Date'].dt.date
            comparison_data_full = output_df_full[output_df_full['Date'] == selected_date].copy()
            
            # Check if we have both "Today" and "X Days Ago" columns
            today_cols = [col for col in comparison_data_full.columns if 'Total Positions carried Today' in col]
            days_ago_cols = [col for col in comparison_data_full.columns if f'Total Positions carried {days_ago_num} Day Ago' in col or f'Total Positions carried {days_ago_num} Days Ago' in col]
            
            if today_cols and days_ago_cols:
                # Create comparison data
                comparison_data = []
                
                # Extract segment names and create comparison
                segments = ['Index Futures', 'Stock Futures', 'Index Call', 'Index Put', 'Stock Call', 'Stock Put']
                
                for client_type in comparison_data_full['CLIENT_TYPE'].unique():
                    client_row = comparison_data_full[comparison_data_full['CLIENT_TYPE'] == client_type].iloc[0]
                    
                    for segment in segments:
                        today_col = f'Total Positions carried Today {segment}'
                        days_ago_col = None
                        
                        # Find matching days ago column
                        for col in days_ago_cols:
                            if segment in col:
                                days_ago_col = col
                                break
                        
                        if today_col in comparison_data_full.columns and days_ago_col:
                            today_val = safe_float(client_row.get(today_col, 0))
                            days_ago_val = safe_float(client_row.get(days_ago_col, 0))
                            change = today_val - days_ago_val
                            change_pct = (change / days_ago_val * 100) if days_ago_val != 0 else 0
                            
                            comparison_data.append({
                                'CLIENT_TYPE': client_type,
                                'Segment': segment,
                                'Today': today_val,
                                f'{selected_days_ago}': days_ago_val,
                                'Change': change,
                                'Change %': change_pct
                            })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    # Visualization 1: Side-by-side bar chart comparing Today vs X Days Ago
                    st.markdown("**6.1. Total Positions: Today vs " + selected_days_ago + "**")
                    
                    # Create grouped bar chart for each client type
                    for client_type in comp_df['CLIENT_TYPE'].unique():
                        client_comp = comp_df[comp_df['CLIENT_TYPE'] == client_type]
                        
                        fig = go.Figure()
                        
                        # Add Today bars
                        fig.add_trace(go.Bar(
                            name='Today',
                            x=client_comp['Segment'],
                            y=client_comp['Today'],
                            marker_color='#2E86AB',
                            text=client_comp['Today'].apply(lambda x: f'{x:,.0f}'),
                            textposition='outside'
                        ))
                        
                        # Add X Days Ago bars
                        fig.add_trace(go.Bar(
                            name=selected_days_ago,
                            x=client_comp['Segment'],
                            y=client_comp[f'{selected_days_ago}'],
                            marker_color='#A23B72',
                            text=client_comp[f'{selected_days_ago}'].apply(lambda x: f'{x:,.0f}'),
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title=f'{client_type} - Total Positions Comparison',
                            xaxis_title='Segment',
                            yaxis_title='Total Positions',
                            barmode='group',
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualization 2: Change/Delta Chart
                    st.markdown("**6.2. Change Analysis (Today - " + selected_days_ago + ")**")
                    
                    # Create heatmap of changes
                    pivot_change = comp_df.pivot(index='CLIENT_TYPE', columns='Segment', values='Change')
                    
                    fig = px.imshow(
                        pivot_change,
                        labels=dict(x="Segment", y="Client Type", color="Change"),
                        x=pivot_change.columns,
                        y=pivot_change.index,
                        color_continuous_scale='RdYlGn',
                        aspect="auto",
                        title='Position Change Heatmap (Green = Increase, Red = Decrease)'
                    )
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualization 3: Percentage Change by Segment
                    st.markdown("**6.3. Percentage Change by Segment**")
                    
                    fig = px.bar(
                        comp_df,
                        x='Segment',
                        y='Change %',
                        color='CLIENT_TYPE',
                        barmode='group',
                        title=f'Percentage Change: Today vs {selected_days_ago}',
                        labels={'Change %': 'Percentage Change (%)', 'Segment': 'Segment'}
                    )
                    
                    fig.update_layout(
                        xaxis_title='Segment',
                        yaxis_title='Percentage Change (%)',
                        height=400
                    )
                    
                    # Add horizontal line at 0
                    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualization 4: Summary Table
                    st.markdown("**6.4. Detailed Comparison Table**")
                    
                    # Create a styled comparison table
                    summary_table = comp_df.pivot_table(
                        index=['CLIENT_TYPE', 'Segment'],
                        values=['Today', f'{selected_days_ago}', 'Change', 'Change %'],
                        aggfunc='first'
                    ).reset_index()
                    
                    # Format the table
                    summary_table['Today'] = summary_table['Today'].apply(lambda x: f'{x:,.0f}')
                    summary_table[f'{selected_days_ago}'] = summary_table[f'{selected_days_ago}'].apply(lambda x: f'{x:,.0f}')
                    summary_table['Change'] = summary_table['Change'].apply(lambda x: f'{x:+,.0f}')
                    summary_table['Change %'] = summary_table['Change %'].apply(lambda x: f'{x:+.2f}%')
                    
                    st.dataframe(summary_table, use_container_width=True, hide_index=True)
                    
                    # Visualization 5: Overall Change Direction by Client Type
                    st.markdown("**6.5. Overall Change Summary by Client Type**")
                    
                    # Calculate total change for each client type
                    client_summary = comp_df.groupby('CLIENT_TYPE').agg({
                        'Change': 'sum',
                        'Today': 'sum',
                        f'{selected_days_ago}': 'sum'
                    }).reset_index()
                    
                    client_summary['Change %'] = (client_summary['Change'] / client_summary[f'{selected_days_ago}'] * 100).fillna(0)
                    
                    # Create gauge-like indicators
                    cols = st.columns(len(client_summary))
                    
                    for idx, (_, row) in enumerate(client_summary.iterrows()):
                        with cols[idx]:
                            change_val = row['Change']
                            change_pct = row['Change %']
                            
                            # Determine color based on change
                            color = 'green' if change_val > 0 else 'red' if change_val < 0 else 'gray'
                            
                            fig = go.Figure(go.Indicator(
                                mode="number+delta",
                                value=change_val,
                                delta={
                                    'reference': 0,
                                    'valueformat': '.0f',
                                    'increasing': {'color': 'green'},
                                    'decreasing': {'color': 'red'}
                                },
                                title={'text': f"<b>{row['CLIENT_TYPE']}</b><br>Change: {change_pct:.2f}%"},
                                number={'valueformat': ',.0f', 'font': {'color': color, 'size': 60}}
                            ))
                            
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)


