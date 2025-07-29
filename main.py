import yfinance as yf
import numpy as np
import pandas as pd

def compute_features(data, shares) -> pd.DataFrame:
    """Calculate addtional features from a raw dataframe fetched from yf.

    Derive 9 additional features from a raw dataframe obtained from yfinance, including a
    target column "Tomorrow Return" that will be used to train a ML model. Values of this
    column is produced by shifting all returns backwards by one day. 

    Args:
        data: yf dataframe consisting of five columns ['Open', 'High', 'Close', 'Low', 'Volume']
        shares: number of outstanding shares 
    """
    data['Amount'] = data['Close'] * data['Volume']
    data['Relative Volume(20d)'] = data['Volume'] / data['Volume'].rolling(20).mean()

    if not np.isnan(shares):
        data['Turnover'] = data['Volume'] / shares
    else:
        data['Turnover'] = np.nan

     # Compute daily price changes
    data['Open Change'] = data['Open'].pct_change()
    data['High Change'] = data['High'].pct_change()
    data['Close Change'] = data['Close'].pct_change()
    data['Low Change'] = data['Low'].pct_change()
    
    # Create price prediction target
    data['Today Return'] = data['Close Change']
    data['Tomorrow Return'] = data['Today Return'].shift(-1)

    return data
