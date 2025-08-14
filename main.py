import pandas as pd
import numpy as np

def winsorize(col, mode=2, upper=0.99, low=0.01, z=5):
    """Mitigate impact of outliers by applying winsorization to a data column.

    Trims extreme values in a dataframe column by appling one of two strategies: quantile winsorization and
    Median Absolute Deviation(MAD) winsorization. In the former, a default upper bound of 99 percentile and
    lower bound of 1 percentile is employed to clip outliers; while the latter applies fivefold MAD values in
    either direction to clip values. This method defaults to the latter mode.

    Args:
        col(str): dataframe column to clip
        mode: type of winsorization strategy. Defaults to MAD based winsorization.
        upper: upper bound of quantile winsorization.
        lower: lower bound of quantile winsorzation.
        z: number of MADs that delimits the range of MAD based strategy.

    Returns:

    """
    if mode == 1:
        upper_bound = col.quantile(upper)
        lower_bound = col.quantile(low)
    else:
        median = col.median()
        mad = col.median(np.abs(col - median))
        upper_bound = median + z * mad
        lower_bound = median - z * mad
    col = col.clip(lower_bound, upper_bound)
    return col