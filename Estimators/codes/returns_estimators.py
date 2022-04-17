import numpy as np
import pandas as pd

class ReturnsEstimators:

    def __init__(self):
        pass

    @staticmethod
    def calculate_mean_historical_returns(asset_prices, resample_by=None, frequency=252):
        asset_prices = pd.DataFrame(asset_prices) if not isinstance(asset_prices, pd.DataFrame) else asset_prices
        returns = ReturnsEstimators().calculate_returns(asset_prices, resample_by)
        return returns.mean() * frequency

    @staticmethod
    def calculate_exponential_historical_returns(asset_prices, resample_by=None, frequency=252, span=500):
        asset_prices = pd.DataFrame(asset_prices) if not isinstance(asset_prices, pd.DataFrame) else asset_prices
        returns = ReturnsEstimators().calculate_returns(asset_prices, resample_by)
        return returns.ewm(span=span).mean().iloc[-1] * frequency

    @staticmethod
    def calculate_returns(asset_prices, resample_by=None):
        asset_prices = asset_prices.resample(resample_by).sum() if resample_by != None else asset_prices
        return asset_prices.pct_change().dropna(how="all")