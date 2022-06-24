import numpy as np
import pandas as pd

def remove_nan(data, which="col"):
    
    if isinstance(data, np.ndarray):
        axis = 0 if which=="col" else 1 # 0 drops cols, 1 drops rows
        data = data[np.isnan(data.any(axis=axis))]
    
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        axis = 1 if which=="col" else 0 # 1 drops cols, 0 drops rows
        data = data.dropna(axis=axis)
        
    return data


def fill_nan(data, idx, idx_label=None, which="col"):
    
    data_nan = np.array(data)
        
    if which.startswith("row"):
        for i in idx:
            data_nan = np.insert(data_nan, i, np.zeros((1,data.shape[1])), axis=0)
            data_nan[i,:] = np.nan
                       
    elif which.startswith("col"):
        for i in idx:
            data_nan = np.insert(data_nan, i, np.zeros((1,data.shape[0])), axis=1)
            data_nan[:,i] = np.nan
    
    if isinstance(data, pd.DataFrame):
        
        if idx_label is None:
            idx_label = ["nan"]*len(idx)
            
        if which.startswith("row"):
            index = list(data.index)
            for i_label, i in enumerate(idx):
                index[i:i] = [idx_label[i_label]]
            data_nan = pd.DataFrame(data=data_nan, index=index, columns=data.columns)
            
        elif which.startswith("col"):
            columns = list(data.columns)
            for i_label, i in enumerate(idx):
                columns[i:i] = [idx_label[i_label]]
            data_nan = pd.DataFrame(data=data_nan, index=data.index, columns=columns)

    return data_nan