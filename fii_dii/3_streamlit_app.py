import pandas as pd
import streamlit as st
import os
from config import PATH_DATA_FILES, PATH_RAW_FILE, PATH_STREAMLIT_DOWNLOAD_FILE

#! Load the data
df = pd.read_csv(PATH_RAW_FILE)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df_sorted = df.sort_values(by=['Date', 'CLIENT_TYPE'])

#! Create a function to process data
def process_data(df_sorted):
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
        'Total Positions carried Today Stock Futures',
        'Total Positions carried 1 Day Ago Stock Futures',
        'Total Positions carried 2 Days Ago Stock Futures',
        'Total Positions carried Today Index Call',
        'Total Positions carried 1 Day Ago Index Call',
        'Total Positions carried 2 Days Ago Index Call',
        'Total Positions carried Today Index Put',
        'Total Positions carried 1 Day Ago Index Put',
        'Total Positions carried 2 Days Ago Index Put',
        'Total Positions carried Today Stock Call',
        'Total Positions carried 1 Day Ago Stock Call',
        'Total Positions carried 2 Days Ago Stock Call',
        'Total Positions carried Today Stock Put',
        'Total Positions carried 1 Day Ago Stock Put',
        'Total Positions carried 2 Days Ago Stock Put',
    ])

    for client_type in df_sorted['CLIENT_TYPE'].unique():
        if client_type == 'Total':
            continue

        client_data = df_sorted[df_sorted['CLIENT_TYPE'] == client_type]
        temp_changes = []

        for date in client_data['Date'].unique():
            today_row = client_data[client_data['Date'] == date]
            
            if date == client_data['Date'].min():
                changes_row = {  # Initialization of the row
                    'Date': date,
                    'CLIENT_TYPE': client_type,
                    # Add your field processing logic here
                    # For example, you can set all fields to None or zeros for the first row
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
                    'Total Positions carried Today Stock Futures': None,
                    'Total Positions carried 1 Day Ago Stock Futures': None,
                    'Total Positions carried 2 Days Ago Stock Futures': None,
                    'Total Positions carried Today Index Call': None,
                    'Total Positions carried 1 Day Ago Index Call': None,
                    'Total Positions carried 2 Days Ago Index Call': None,
                    'Total Positions carried Today Index Put': None,
                    'Total Positions carried 1 Day Ago Index Put': None,
                    'Total Positions carried 2 Days Ago Index Put': None,
                    'Total Positions carried Today Stock Call': None,
                    'Total Positions carried 1 Day Ago Stock Call': None,
                    'Total Positions carried 2 Days Ago Stock Call': None,
                    'Total Positions carried Today Stock Put': None,
                    'Total Positions carried 1 Day Ago Stock Put': None,
                    'Total Positions carried 2 Days Ago Stock Put': None,
                }
            else:
                prev_dates = client_data[client_data['Date'] < date]['Date'].unique()
                if len(prev_dates) >= 2:
                    prev_date_1 = prev_dates[-1]
                    prev_date_2 = prev_dates[-2]
                else:
                    prev_date_1 = prev_dates[0] if len(prev_dates) > 0 else client_data['Date'].min()
                    prev_date_2 = prev_date_1

                prev_row_1 = client_data[client_data['Date'] == prev_date_1]
                prev_row_2 = client_data[client_data['Date'] == prev_date_2]
                
                # Calculate changes (assuming all fields are numeric)
                changes_row = {
                    'Date': date,
                    'CLIENT_TYPE': client_type,
                    # Calculate the changes between current and previous rows
                    'Future Index Long': today_row['Future Index Long'].values[0] - prev_row_1['Future Index Long'].values[0],
                    # Continue for other columns
                    # Include logic to calculate other columns as required
                    'Future Index Short': today_row['Future Index Short'].values[0] - prev_row_1['Future Index Short'].values[0],
                    'Future Stock Long': today_row['Future Stock Long'].values[0] - prev_row_1['Future Stock Long'].values[0],
                    'Future Stock Short': today_row['Future Stock Short'].values[0] - prev_row_1['Future Stock Short'].values[0],
                    # Continue for all other fields
                }

            temp_changes.append(changes_row)

        client_output_df = pd.DataFrame(temp_changes)
        output_df = pd.concat([output_df, client_output_df], ignore_index=True)

    output_df = output_df[output_df['CLIENT_TYPE'] != 'TOTAL']
    return output_df

#! Streamlit UI
st.title('Open Interest Data')
st.write("Filter the data by selecting a date.")

#! Date picker for selecting the filter date
selected_date = st.date_input("Select a Date", min_value=df['Date'].min(), max_value=df['Date'].max(), value=df['Date'].max())

#! Process data and filter based on selected date
filtered_data = process_data(df_sorted)
filtered_data = filtered_data[filtered_data['Date'] == pd.to_datetime(selected_date)]

#! Display filtered data in a table
st.write(filtered_data)

#! Allow the user to download the filtered data as a CSV
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name=PATH_STREAMLIT_DOWNLOAD_FILE,
    mime='text/csv'
)
