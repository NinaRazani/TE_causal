from socket import close
from typing import Counter
from datatable import last
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import talib


from collect_data import concat, extend_concat, fred_down, h_forex_data, yfin_down, apply_talib_indicator, to_supervised_multi
from transfer_entropy import transfer_ent
# from encoder import create_transformer_encoder

##download data 
#dataset
major_fx = ['AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X']
# macto_fred_ids = ['M2SL', 'RBUSBIS', 'REAINTRATREARAT10Y', 'inflation']
# macro_yahoo_finance = [bac:BAC, gold:GC=F, wti:CL=F, brent:BZ=F, gas:NG=F] 
# indicators = [SMA, MACD, RSI] 

def hour_prepare_features(forex_id, start_date, end_date, flow=True): 
    #daily
    #  h_data = yfin_down(forex_id, start_date, end_date,'1d', False) 
    #hourly
     h_data = h_forex_data(forex_id) 
     bac_df = yfin_down('BAC', start_date, end_date,'1d', True)  
     gold_df = yfin_down('GC=F', start_date, end_date,'1d', True)  
     WTI_df = yfin_down('CL=F', start_date, end_date,'1d', True) 
     brent_df = yfin_down('BZ=F', start_date, end_date,'1d', True) 
     gas_df = yfin_down('NG=F', start_date, end_date,'1d', True) 

     currency_dfs = {}
     for frx in major_fx:
         if frx != forex_id:
             currency_dfs[frx] = h_forex_data(frx)
    
     real_df = fred_down('RBUSBIS', "date", "rate", start_date, end_date, True) 
     MS_df = fred_down('M2SL', "date", "rate", start_date, end_date, True) 
     interest_df = fred_down('REAINTRATREARAT10Y', "date", "rate", start_date, end_date, True) 
     inflation_rate_df = fred_down('T10YIE', "date", "rate", start_date, end_date, True) 
     
    #  # this lines done because the inflation is daily 
     h_data = h_data.set_index('date')
     inflation_rate_df = inflation_rate_df.set_index('date') 
     result = h_data.join(inflation_rate_df, how='inner') 
     h_data = result[h_data.columns].reset_index()
     inflation_rate_df = result[inflation_rate_df.columns].reset_index() 
    #  # ##
     sma_df = apply_talib_indicator(h_data, talib.SMA, 'Close', 'SMA_20', timeperiod=20)   # type: ignore  
    
     macd_df = apply_talib_indicator(h_data, talib.MACD, 'Close', ('MACD', 'MACD_signal', 'MACD_hist'),  # type: ignore 
                                     fastperiod=12, slowperiod=26, signalperiod=9)  
     rsi_df = apply_talib_indicator(h_data, talib.RSI, 'Close', 'RSI_14', timeperiod=14) #type: ignore 
     def normalize_rsi(rsi):
         return (rsi - 50) / 50
     rsi_df['RSI_14'] = rsi_df['RSI_14'].apply(normalize_rsi) 
     
    # # this lines done because the inflation is daily 
    # #  h_data = h_data.set_index('date')
    # #  inflation_rate_df = inflation_rate_df.set_index('date') 
    # #  result = h_data.join(inflation_rate_df, how='inner') 
    # #  h_data = result[h_data.columns].reset_index()
    # #  inflation_rate_df = result[inflation_rate_df.columns].reset_index()  
    # #  # ##
     
     if flow==True:
        #  collect transfer entropy
         yah_bac = concat(bac_df, h_data)
         te_bac = transfer_ent(yah_bac, 264 )
         yah_gold = concat(gold_df, h_data) 
         te_gold = transfer_ent(yah_gold, 264) 
         yah_wti = concat(WTI_df, h_data)
         te_wti = transfer_ent(yah_wti, 264 ) 
         yah_brent = concat(brent_df, h_data)
         te_brent = transfer_ent(yah_brent, 264 ) 
         yah_gas = concat(gas_df, h_data) 
         te_gas = transfer_ent(yah_gas, 264 )  

         #between currencies
         cur_concat_dfs = {}
         cur_entropy_dfs = {}
         for frx in major_fx:
             if frx != forex_id:
                 cur_concat_dfs[frx] = concat( currency_dfs[frx], h_data)  
                 cur_entropy_dfs[frx] = transfer_ent(cur_concat_dfs[frx], 264 )
         
         merg_ms_yf_df = extend_concat(MS_df, h_data, 'M')
         te_ms = transfer_ent(merg_ms_yf_df, 264 ) 
         merg_re_yf_df = extend_concat(real_df, h_data, 'M')
         te_re = transfer_ent(merg_re_yf_df, 264 )
         merg_ir_yf_df = extend_concat(interest_df, h_data, 'M') 
         te_ir = transfer_ent(merg_ir_yf_df, 264 )
         merg_inf_yf_df = extend_concat(inflation_rate_df, h_data)
         te_inf = transfer_ent(merg_inf_yf_df, 264 ) 
         
         #add datetime
         h_data.reset_index(inplace=True)
         h_data['dayofweek'] = h_data['date'].dt.day_of_week
         h_data['date'] = pd.to_datetime(h_data['date'], format='%Y.%m')
         h_data['yearmonth'] = h_data['date'].dt.month
         h_data['month'] = h_data['date'].dt.month
         
         # rename columns and reset index
         merg_ms_yf_df.rename(columns={"rate": "rate_ms"}, inplace=True)
         merg_re_yf_df.rename(columns={"rate":"rate_re"}, inplace=True)
         merg_ir_yf_df.rename(columns={"rate":"rate_ir"}, inplace=True) 
         merg_inf_yf_df.rename(columns={"rate":"rate_inf"}, inplace=True) 
         
         te_ms.rename(columns={"mean_flow":"ms_flow"}, inplace=True)
         te_re.rename(columns={"mean_flow":"re_flow"}, inplace=True)
         te_ir.rename(columns={"mean_flow":"ir_flow"}, inplace=True)
         te_inf.rename(columns={"mean_flow":"inf_flow"}, inplace=True)
         te_bac.rename(columns={"mean_flow":"bac_flow"}, inplace=True)
         te_gold.rename(columns={"mean_flow":"gold_flow"}, inplace=True)
         te_wti.rename(columns={"mean_flow":"wti_flow"}, inplace=True)
         te_brent.rename(columns={"mean_flow":"brent_flow"}, inplace=True)
         te_gas.rename(columns={"mean_flow":"gas_flow"}, inplace=True)

         
         for name, df in cur_entropy_dfs.items():
             df.rename(columns={"mean_flow":name}, inplace=True)

         yah_bac.rename(columns={col.strip(): "bac" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_brent.rename(columns={col.strip(): "brent" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_gas.rename(columns={col.strip(): "gas" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_gold.rename(columns={col.strip(): "gold" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_wti.rename(columns={col.strip(): "wti" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         
         ##
         
         te_ms = te_ms.reset_index(drop=True)
         te_re = te_re.reset_index(drop=True)
         te_ir= te_ir.reset_index(drop=True)
         te_inf = te_inf.reset_index(drop=True) 
         
         for name, df in cur_entropy_dfs.items(): 
             cur_entropy_dfs[name] = df.reset_index(drop=True)

         te_bac = te_bac.reset_index(drop=True)
         te_gold = te_gold.reset_index(drop=True)
         te_wti = te_wti.reset_index(drop=True)
         te_brent = te_brent.reset_index(drop=True) 
         te_gas = te_gas.reset_index(drop=True)

         for name, df in cur_entropy_dfs.items():
             df = df.reset_index(drop=True)
             clean_name = name.replace('=', '').replace('X', '').replace('#', '')
             clean_name += "flow"
             df.rename(columns={name:clean_name}, inplace=True)
         
         # #concatenate dataframes 
         row_to_remove = merg_ms_yf_df.shape[0] - te_bac.shape[0] 
         
         last_df = pd.concat([te_ms['ms_flow'], te_re['re_flow'],te_ir['ir_flow'], te_inf['inf_flow'],
                              te_bac['bac_flow'], te_gold['gold_flow'], te_wti['wti_flow'], te_brent['brent_flow'], te_gas['gas_flow']],  axis=1)
            
         for name, df in cur_entropy_dfs.items():
             last_df = pd.concat([last_df, df[name]], axis=1)

         last_df.reset_index(inplace=True)
         last_df.drop(['index'], axis=1, inplace=True) 

         dfs_to_add = [merg_re_yf_df, merg_ir_yf_df, merg_inf_yf_df, merg_ms_yf_df, yah_bac, yah_gold, yah_wti, yah_brent, yah_gas]
         dfs_to_add2 = [sma_df, macd_df, rsi_df] 
         
         for i, df in enumerate(dfs_to_add, 1): 
        
             df = df.iloc[row_to_remove:, :]  
             print(df.shape)
             print(last_df.shape)
             column_name = df.columns[0]  # Get the name of the first column 
             last_df[column_name] = df[column_name].values 
         
         
         for i, df in enumerate(dfs_to_add2, 1): 
             
             df = df.iloc[row_to_remove:, :]  
             if df is sma_df:
                 sma_df['SMA_20'] = (sma_df['SMA_20']-sma_df['SMA_20'].min()) / (sma_df['SMA_20'].max()-sma_df['SMA_20'].min())
             print(df.head(26))
             print(df.tail(26))
             print(df.shape)
             print(last_df.head(26))
             print(last_df.tail(26))  
             print(last_df.shape) 
             column_name = df.columns[2]  # Get the name of the first column
             last_df[column_name] = df[column_name].values 
         
         m = []
         s = []
        #  for i in range(len(h_data)): 
        #      close_column = h_data.iloc[:i, 1].values
        #      S = np.std(close_column)  # type: ignore 
        #      s.append(S)
        #      M = np.mean(close_column) # type: ignore   
        #      m.append(M) 

         for i in range(len(h_data)):
             if i<=22:
                 s.append(0)
                 m.append(0)
             else:
                 close_column = h_data.iloc[i-22:i, 1].values 
                 S = np.std(close_column)  # type: ignore 
                 s.append(S)
                 M = np.mean(close_column) # type: ignore   
                 m.append(M) 
    
         h_data['Std'] = s
         h_data['Mean'] = m
         
         h_data = h_data.iloc[row_to_remove:, :] 
         h_data.reset_index(inplace=True)
         
         last_df['dayweek'] = h_data['dayofweek']
         last_df['monthyear'] = h_data['yearmonth']
    
         last_df['ret_Close'] = np.log(abs(h_data['Close'])) - np.log(abs(h_data['Close'].shift(1)))
         last_df = last_df.fillna(method='bfill') # type: ignore
    
         last_df['Mean'] = h_data['Mean']
         last_df['Std'] = h_data['Std']
         last_df['target'] = h_data['Close']
     else:
         yah_bac = concat(bac_df, h_data)
         yah_gold = concat(gold_df, h_data)
         yah_wti = concat(WTI_df, h_data)
         yah_brent = concat(brent_df, h_data)
         yah_gas = concat(gas_df, h_data)     
         merg_ms_yf_df = extend_concat(MS_df, h_data, 'M')
         merg_re_yf_df = extend_concat(real_df, h_data, 'M')
         merg_ir_yf_df = extend_concat(interest_df, h_data, 'M') 
         merg_inf_yf_df = extend_concat(inflation_rate_df, h_data)
         
         #add datetime
         h_data.reset_index(inplace=True)
         h_data['dayofweek'] = h_data['date'].dt.day_of_week
         h_data['date'] = pd.to_datetime(h_data['date'], format='%Y.%m')
         h_data['yearmonth'] = h_data['date'].dt.month
         h_data['month'] = h_data['date'].dt.month
         
         # rename columns and reset index
         merg_ms_yf_df.rename(columns={"rate": "rate_ms"}, inplace=True)
         merg_re_yf_df.rename(columns={"rate":"rate_re"}, inplace=True)
         merg_ir_yf_df.rename(columns={"rate":"rate_ir"}, inplace=True) 
         merg_inf_yf_df.rename(columns={"rate":"rate_inf"}, inplace=True) 
         
         yah_bac.rename(columns={col.strip(): "bac" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_brent.rename(columns={col.strip(): "brent" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_gas.rename(columns={col.strip(): "gas" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_gold.rename(columns={col.strip(): "gold" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         yah_wti.rename(columns={col.strip(): "wti" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
         
         # #concatenate dataframes 
        #  row_to_remove = merg_ms_yf_df.shape[0] - te_ms.shape[0] 
         
         dfs_to_add = [merg_re_yf_df, merg_ir_yf_df, merg_inf_yf_df, merg_ms_yf_df, yah_bac, yah_gold, yah_wti, yah_brent, yah_gas] 
         dfs_to_add2 = [sma_df, macd_df, rsi_df]
         last_df = pd.DataFrame() 
         for i, df in enumerate(dfs_to_add, 1):   
             column_name = df.columns[0]  # Get the name of the first column
             last_df[column_name] = df[column_name].values 
         
         
         for i, df in enumerate(dfs_to_add2, 1): 
             column_name = df.columns[2]  # Get the name of the first column
             last_df[column_name] = df[column_name].values 
         
         m = []
         s = []
        #  for i in range(len(h_data)): 
        #      close_column = h_data.iloc[:i, 1].values
        #      S = np.std(close_column)  # type: ignore 
        #      s.append(S)
        #      M = np.mean(close_column) # type: ignore   
        #      m.append(M) 

         for i in range(len(h_data)):
            if i<=22:
                s.append(0)
                m.append(0)
            else:
                 close_column = h_data.iloc[i-22:i, 1].values 
                 S = np.std(close_column)  # type: ignore 
                 s.append(S)
                 M = np.mean(close_column) # type: ignore   
                 m.append(M) 
         
         h_data['Std'] = s
         h_data['Mean'] = m
         
        #  h_data = h_data.iloc[row_to_remove:, :] 
        #  h_data.reset_index(inplace=True) 
         
         last_df['dayweek'] = h_data['dayofweek']
         last_df['monthyear'] = h_data['yearmonth']
    
        #  last_df['ret_Close'] = np.log(abs(h_data['Close'])) - np.log(abs(h_data['Close'].shift(1))) 
         last_df = last_df.fillna(method='bfill') # type: ignore
    
         last_df['Mean'] = h_data['Mean']
         last_df['Std'] = h_data['Std']
         last_df['target'] = h_data['Close']
         last_df = last_df.dropna() 

     return last_df 