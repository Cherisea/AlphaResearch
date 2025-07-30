import yfinance as yf
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def compute_features(data, shares) -> pd.DataFrame:
    """Calculate addtional features from a raw dataframe fetched from yf.

    Derive 9 additional features from a raw dataframe obtained from yfinance, including a
    target column "Tomorrow Return" that will be used to train a ML model. Values of this
    column is produced by shifting all returns backwards by one day. 

    Args:
        data(pd.DataFrame): yf dataframe consisting of five columns ['Open', 'High', 'Close', 'Low', 'Volume']
        shares(int): number of outstanding shares 
    
    Returns:
        data(pd.DataFrame): dataframe with addtional features 
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

def prepare_data(ticker_list, start_date, end_date):
    """Create a data file in a format required by a ML model.

    This method downloads raw market data from yfinance, adds additional derived features and
    store it in a file format for use by a model.

    Args:
        ticker_list(list): a list of stock tickers (U.S market)
        start_date(str): start date of market data to pull
        end_date(str): end date of market data to pull
    
    Returns:
        all_data(list): a list of dataframes, where each one contains an enhanced dataframe
            for each stock
    """
    all_data = []

    for ticker in ticker_list:
        raw_df = yf.download(ticker, start_date, end_date)

        # Skip if can't retrieve market data
        if raw_df.empty:
            continue
        
        # Create a deep copy of raw df with only columns we need
        data = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Fetch number of shares outstanding
        ticker_info = yf.Ticker(ticker).info
        shares = ticker_info.get('sharesOutstanding', np.nan)
        
        # Process raw dataframe
        df = compute_features(data, shares)

        # Drop nested index and remaining index names
        df.columns = df.columns.droplevel(1)
        df.columns.name = None

        # Replace inner index with a separate column
        df['Ticker'] = ticker

        # Reverts index back to integer-based index
        df.reset_index(inplace=True)

        all_data.append(df)
    
    return all_data

def save_to_file(all_data, file_path):
    """Save a list of dataframes to a csv file.

    Args:
        all_data(list): a list of dataframes of market data
        file_path(str/Path)
    """
    # Stack dataframes vertically and reset index
    df = pd.concat(all_data, ignore_index=True)

    # Drop rows with invalid value for the target column
    df.dropna(subset=['Tomorrow Return'], inplace=True)
    df = df[np.isfinite(df['Tomorrow Return'])]

    df = df[['Ticker', 'Date', 'Relative Volume', 'Amount', 'Turnover',
         'Open Change', 'High Change', 'Low Change', 'Close Change',
         'Tomorrow Return']]

    df.to_csv(file_path, index=False)
    print(f"Market data saved to {file_path}")

def get_tickers():
    """Retrieve a list of S&P500 tickers from Wikipedia.

    Returns:
    
    """
    wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(wiki_url)

    assert response.status_code == 200, "Server didn't send a valid response!"

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    
    tickers = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if cells:
            ticker = cells[0].text.strip()
            if ticker:
                tickers.append(ticker)
    
    return tickers

def main():
    start_date = '2022-01-01'
    end_date = '2025-01-01'
    ticker_list = get_tickers()
    data = prepare_data(ticker_list, start_date, end_date)
    save_to_file(data, 'df.csv')

if __name__ == '__main__':
    main()
    
