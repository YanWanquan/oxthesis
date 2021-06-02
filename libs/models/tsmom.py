# --- --- ---
# tsmom.py
# Sven Giegerich / 03.05.2021
# --- --- ---

from os import stat
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import libs.utils as utils


class MomentumInterface(ABC):

    @abstractmethod
    def calc_signal(self, prices):
        pass

    @abstractmethod
    def calc_position(signals):
        pass

    def calc_strategy_returns(self, df):
        prices = df.xs('prs', axis=1, level=1, drop_level=True)
        scaled_rts = df.xs('rts_scaled', axis=1, level=1, drop_level=True)

        signals = self.calc_signal(prices)
        positions = self.calc_position(signals)
        returns = utils.calc_strategy_returns(
            positions=positions, realized_returns=scaled_rts)

        return (positions, returns)


class LongOnlyStrategy(MomentumInterface):
    name = 'long'

    @staticmethod
    def calc_signal(prices):
        return pd.DataFrame(np.ones(prices.shape), index=prices.index, columns=prices.columns)

    @staticmethod
    def calc_position(signals):
        return pd.DataFrame(np.ones(signals.shape), index=signals.index, columns=signals.columns)


class BasicMomentumStrategy(MomentumInterface):
    """
    Returns signals based on the classic Time Series Momentum (TSMOM) strategy,
    Moskowitz, T.J., Ooi, Y.H. and Pedersen, L.H., 2012. 
    Time series momentum. Journal of Financial Economics, 104(2), pp.228-250.
    """
    name = 'tsmom'

    def __init__(self, lookback=252):
        self.lookback = lookback

    def calc_signal(self, prices):
        """
        Args:
            df (pd.Dataframe): dataframe of (raw) prices (dim: T x instruments)
        """
        rts = utils.calc_returns_df(prices, offset=self.lookback)
        return rts.apply(np.sign)

    @staticmethod
    def calc_position(signals):
        """
        Args:
            signals (pd.Dataframe): dataframe of signals (dim: T x instruments)
        """
        return signals.apply(np.sign)


class MACDStrategy(MomentumInterface):
    """
    Returns signals based on,
    Baz, J., Granger, N., Harvey, C.R., Le Roux, N. and Rattray, S. 2015. 
    Dissecting investment strategies in the cross section and time series. Available at SSRN 2695101.
    """

    def __init__(self, trd_vol_win=252, sgn_vol_win=63, trd_comb=None):
        """
        Args
            trd_vol_win (252): volatility window for the trend estimation
            sgn_vol_win (63): volatility window for the signal function
            trd_comb (None): ...
        """
        self.trd_vol_win = trd_vol_win
        self.sgn_vol_win = sgn_vol_win

        if trd_comb is None:
            self.trd_comb = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trd_comb = trd_comb

    def calc_signal_scale(self, prices, short_win, long_win):
        """Compute the signal for one specific time scale"""
        short_trd = prices.ewm(halflife=self.get_halflife(short_win)).mean()
        long_trd = prices.ewm(halflife=self.get_halflife(long_win)).mean()
        vol_prices = prices.rolling(
            self.sgn_vol_win).std().fillna(method='bfill')

        macd = short_trd - long_trd
        q = macd / vol_prices
        trd = q / q.rolling(self.trd_vol_win).std().fillna(method='bfill')

        return trd

    def calc_signal(self, prices):
        sgns = None

        # average multiple signals w diff time scales
        for short_win, long_win in self.trd_comb:
            sgn = self.calc_signal_scale(prices, short_win, long_win)

            if sgns is None:
                sgns = self.scale_signal(sgn)
            else:
                sgns += self.scale_signal(sgn)

        return sgns / len(self.trd_comb)

    def calc_position(signals):
        raise NotImplementedError("To be done!")
        pass

    @staticmethod
    def scale_signal(z):
        return z * np.exp(-z**2 / 4) / 0.89

    @staticmethod
    def get_halflife(s):
        """
        Args
            s: time scale 
        """
        return np.log(0.5) / np.log(1 - 1 / s)
