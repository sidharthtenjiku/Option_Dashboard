import pandas as pd
import os
from config import PATH_DATA_FILES, PATH_RAW_FILE, PATH_OUTPUT_FILE
import warnings
warnings.filterwarnings("ignore")

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