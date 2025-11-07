import os

#! Base data folder
PATH_DATA_FILES = os.path.join('C:\\Users\\ISer\\Desktop\\ai_ml dataset\\fii_dii', 'data')

#! Common file paths
PATH_RAW_FILE = os.path.join(PATH_DATA_FILES, '1_participants_raw_oi_nse.csv')
PATH_OUTPUT_FILE = os.path.join(PATH_DATA_FILES, '2_net_output_oi_data.csv')
PATH_STREAMLIT_DOWNLOAD_FILE = os.path.join(PATH_DATA_FILES, '3_streamlit_download_filtered_oi_data.csv')

#! Start date from which the data fetching will start till last date present (1_getting_nse_raw_participants.py) 
START_DATE_STR = '04-11-2025'
