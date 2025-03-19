import numpy as np
import pandas as pd
import scipy
from scipy.signal import correlate
import yfinance

from collect_data import yfin_down


# yah_df1 = yfin_down('USDCHF=X', '2004-01-01', '2023-12-31','1d', False)
# yah_df2 = yfin_down('EURUSD=X', '2004-01-01', '2023-12-31','1d', False)

# yah_df1['ret_Close'] = (yah_df1['Close']-yah_df1['Close'].shift(1)) / yah_df1['Close'].shift(1)   #percentage return
# yah_df2['ret_Close'] = (yah_df2['Close']-yah_df2['Close'].shift(1)) / yah_df2['Close'].shift(1)

# yah_df1 = yah_df1.replace([np.inf, -np.inf], np.nan).dropna(subset=['ret_Close'])
# yah_df2 = yah_df2.replace([np.inf, -np.inf], np.nan).dropna(subset=['ret_Close'])

# cross_corr = correlate(yah_df1['ret_Close'], yah_df2['ret_Close'], mode='full')

# print(cross_corr)
# print(len(cross_corr))
# print(yah_df1.shape) 

import numpy as np

def concatenate_without_overlap(arrays, tolerance=1e-6):
    result = arrays[0].tolist()  # Start with the first array as a Python list
    for i in range(1, len(arrays)):
        current_array = arrays[i]
        # Find the overlap by checking from the end of the result to the start of the current array
        overlap_start = 0
        for j in range(1, len(current_array) + 1):
            if np.allclose(result[-j:], current_array[:j], atol=tolerance):
                overlap_start = j
                break
        
        # Append the non-overlapping part of the current array
        result.extend(current_array[overlap_start:].tolist())
    
    return np.array(result)  # Convert back to a NumPy array if needed

# Example usage
arrays = [
    np.random.rand(66),  # Example random data
    np.random.rand(66),  # Ensure realistic test data
    np.random.rand(66),  # Adjust as needed
]
# Simulate overlapping by making each array overlap with the previous
for i in range(1, len(arrays)):
    arrays[i] = np.concatenate((arrays[i - 1][-10:], arrays[i][10:]))

result = concatenate_without_overlap(arrays)
print(len(result))  # Should be correct (488 in your case)
print(result[-1])