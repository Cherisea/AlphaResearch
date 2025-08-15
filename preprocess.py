import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def winsorize(col, mode=2, upper=0.99, low=0.01, z=5):
    """Mitigate impact of outliers by applying winsorization to a data column.

    Trims extreme values in a dataframe column by appling one of two strategies: quantile winsorization and
    Median Absolute Deviation(MAD) winsorization. In the former, a default upper bound of 99 percentile and
    lower bound of 1 percentile is employed to clip outliers; while the latter applies fivefold MAD values in
    either direction to clip values. This method defaults to the latter mode.

    Args:
        col(str): a numeric dataframe column to clip
        mode: type of winsorization strategy. Defaults to MAD based winsorization.
        upper: upper bound of quantile winsorization.
        lower: lower bound of quantile winsorzation.
        z: number of MADs that delimits the range of MAD based strategy.

    Returns:
        col(str): a modified version of original column after applying winsorization
    """
    if mode == 1:
        upper_bound = col.quantile(upper)
        lower_bound = col.quantile(low)
    else:
        median = col.median()
        mad = np.median(np.abs(col - median))
        upper_bound = median + z * mad
        lower_bound = median - z * mad
    col = col.clip(lower_bound, upper_bound)
    return col

def standardize(df, col_list):
    """Standardize a list of numeric columns in a dataframe.
    
    Rescales columns with z-score normalization(standardization). Result data points have a mean of 0 and std of 1.

    Args:
        df(DataFrame): a dataframe object with numeric columns
        col_list(list): a list of numeric columns to be rescaled
    
    Returns:
        df(DataFrame): rescaled dataframe object
    """
    for col in col_list:
        df[col + ' (Standardized)'] = (df[col] - df.groupby('Date')[col].transform('mean')) / df.groupby('Date')[col].transform('std')
        
        # Fix NAN values when both denominator and numerator are 0
        df[col + ' (Standardized)'] = df[col + ' (Standardized)'].fillna(0.0)

        # Fix inf/-inf values when numerator is not 0 but denominator is
        df[col + ' (Standardized)'] = df[col + ' (Standardized)'].replace([np.inf, -np.inf], 0.0)
    
    return df


file_path = "raw_data.csv"
col_list = ['Volume', 'Amount', 'Turnover', 'Open Change', 
            'High Change', 'Low Change', 'Close Change',
            'Tomorrow Return']
df = pd.read_csv(file_path)
df = df.dropna(subset=col_list)

# Apply winsorization
for col in col_list:
    df[col] = df.groupby('Date')[col].transform(winsorize)

# Apply standardization
df = standardize(df, col_list)

df_train, df_test = train_test_split(df, test_size=0.2)
df_train, df_validate = train_test_split(df_train, test_size=0.3)
print(f"Train Data Type: {type(df_train)}, Length: {len(df_train)}")
print(f"Test Data Type: {type(df_test)}, Length: {len(df_test)}")
print(f"Validatation Data Type: {type(df_validate)}, Length: {len(df_validate)}")
