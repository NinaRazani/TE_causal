
import pandas as pd
import numpy as np
import os

from collect_data import concat, h_forex_data


path1 = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly/USDJPY_Candlestick_1_Hour_BID_01.01.2004-31.12.2007.csv"
path2 = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly/USDJPY_Candlestick_1_Hour_BID_01.01.2008-31.12.2011.csv"
path3 = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly/USDJPY_Candlestick_1_Hour_BID_01.01.2012-31.12.2015.csv"
path4 = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly/USDJPY_Candlestick_1_Hour_BID_01.01.2016-31.12.2019.csv"
path5 = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly/USDJPY_Candlestick_1_Hour_BID_01.01.2020-31.12.2023.csv"

h_data1 = pd.read_csv(path1)
h_data2 = pd.read_csv(path2)
h_data3 = pd.read_csv(path3)
h_data4 = pd.read_csv(path4)
h_data5 = pd.read_csv(path5)

combined_USDJPY_df = pd.concat([h_data1, h_data2, h_data3, h_data4, h_data5], ignore_index=True) 
folder_path = r"/mnt/c/Users/DSD_LAB_Razani/Desktop/datasets downloaded/download_forex_dukascopy_swiss/hourly"
file_path = os.path.join(folder_path, 'combined_USDJPY_data.csv')
combined_USDJPY_df.to_csv(file_path, index=False)
print(combined_USDJPY_df.shape)
print(combined_USDJPY_df.head(3))
print(combined_USDJPY_df.tail(3))

# major_fx = ['AUDUSD=X', 'EURUSD=X', 'USDJPY=X', 'USDCHF=X']
# forex_id = 'USDCHF=X'
# h_data = h_forex_data(forex_id) 
# currency_dfs = {}
# for frx in major_fx:
#          if frx != forex_id:
#              currency_dfs[frx] = h_forex_data(frx)

# print(currency_dfs[major_fx[0]])  

# hdata_first = concat( currency_dfs[major_fx[1]], h_data)  
# print(hdata_first.head(2))