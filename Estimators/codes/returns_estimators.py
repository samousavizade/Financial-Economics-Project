import warnings
import numpy as np
import pandas as pd

class ReturnsEstimators:
    """
    This class contains methods for estimating expected returns. A good estimation of the asset expected returns is very important
    for portfolio optimisation problems and so it is necessary to use good estimates of returns and not just rely on
    simple techniques.
    """

    def __init__(self):
        """
        Initialize
        """


        pass

    @staticmethod
    def calculate_mean_historical_returns(asset_prices, resample_by=None, frequency=252):
        """
        Calculates the annualised mean historical returns from asset price data.
        :param asset_prices: (pd.DataFrame) Asset price data
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :param frequency: (int) Average number of observations per year
        :return: (pd.Series) Annualized mean historical returns per asset
        """

        if not isinstance(asset_prices, pd.DataFrame):
            warnings.warn("prices are not in a dataframe", RuntimeWarning)
            asset_prices = pd.DataFrame(asset_prices)
        
        returns = ReturnsEstimators().calculate_returns(asset_prices, resample_by)

        return returns.mean() * frequency


    @staticmethod
    def calculate_exponential_historical_returns(asset_prices, resample_by=None, frequency=252, span=500):
        """
        Calculates the exponentially-weighted annualized mean of historical returns, giving
        higher weight to more recent data.
        :param asset_prices: (pd.DataFrame) Asset price data
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :param frequency: (int) Average number of observations per year
        :param span: (int) Window length to use in pandas ewm function
        :return: (pd.Series) Exponentially-weighted mean of historical returns
        """

        if not isinstance(asset_prices, pd.DataFrame):
            warnings.warn("prices are not in a dataframe", RuntimeWarning)
            asset_prices = pd.DataFrame(asset_prices)
        
        returns = ReturnsEstimators().calculate_returns(asset_prices, resample_by)

        return returns.ewm(span=span).mean().iloc[-1] * frequency

    @staticmethod
    def calculate_returns(asset_prices, resample_by=None):
        """
        Calculates a dataframe of returns from a dataframe of prices.
        :param asset_prices: (pd.DataFrame) Historical asset prices
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :return: (pd.DataFrame) Returns per asset
        """

        if resample_by != None:
            asset_prices = asset_prices.resample(resample_by).sum()

        return asset_prices.pct_change().dropna(how="all")