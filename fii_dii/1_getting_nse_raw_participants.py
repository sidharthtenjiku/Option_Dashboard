import os
import pandas as pd
from nselib import derivatives
from nsetools import Nse
import datetime
import time
from config import PATH_DATA_FILES, PATH_RAW_FILE, START_DATE_STR

#! Initialize the NSE object
nse = Nse()

#! Create the folder if not exists
if not os.path.exists(PATH_DATA_FILES):
    os.makedirs(PATH_DATA_FILES)


#! Functions (Helping for Main Function)

def fetch_and_save_open_interest_data(date):
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

def get_last_saved_date():
    if os.path.exists(PATH_RAW_FILE):
        df = pd.read_csv(PATH_RAW_FILE)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        last_date = df['Date'].max()
        return last_date
    return None 

#! Functions (Main)

def fetch_historical_data(START_DATE_STR):
    '''
        Fetch historical Open Interest (CLIENT|FII|DII|PRO|TOTAL) data from NSE and save it as raw data to the CSV file (1_participants_raw_oi_nse.csv)
        Args:
            START_DATE_STR: Start date in the format 'dd-mm-yyyy'
        Returns:
            None
    '''
    start_date = datetime.datetime.strptime(START_DATE_STR, '%d-%m-%Y')
    end_date = datetime.datetime.now()
    last_saved_date = get_last_saved_date()
    if last_saved_date:
        print(f"Last saved date: {last_saved_date.strftime('%d-%m-%Y')}")
        start_date = last_saved_date + datetime.timedelta(days=1)

    for current_date in date_range(start_date, end_date):
        current_date_str = current_date.strftime('%d-%m-%Y')
        success = fetch_and_save_open_interest_data(current_date_str)
        if success:
            time.sleep(2)  
        else:
            print(f"Skipping {current_date_str} due to no data or error.")

#! Main

fetch_historical_data(START_DATE_STR) 