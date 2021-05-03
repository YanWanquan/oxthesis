# --- --- ---
# tsmom.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import numpy as np
from abc import ABC, abstractmethod

class MomentumInterface(ABC):

    @abstractmethod
    def get_signal(self, prices):
        pass


class LongOnlyStrategy(MomentumInterface):

    def get_signal(self, prices):
        return np.ones(prices.shape)


class BasicMomentumStrategy(MomentumInterface):
    """
    Returns signals based on the classic Time Series Momentum (TSMOM) strategy,
    Moskowitz, T.J., Ooi, Y.H. and Pedersen, L.H., 2012. 
    Time series momentum. Journal of Financial Economics, 104(2), pp.228-250.
    """

    def __init__(self, lookback=252):
        self.lookback = lookback

    def get_signal(self, prices):
        rts_lookback = prices / prices.shift(self.lookback) - 1
        return rts_lookback.apply(np.sign)


class MACDStrategy(MomentumInterface):
    """
    Returns signals based on,
    Baz, J., Granger, N., Harvey, C.R., Le Roux, N. and Rattray, S., 2015. 
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
            self.trd_comb = [(8,24), (16,48), (32,96)]
        else:
            self.trd_comb = trd_comb

    def calc_signal_scale(self, prices, short_win, long_win):
        """Compute the signal for one specific time scale"""
        short_trd = prices.ewm(halflife=get_halflife(short_win)).mean()
        long_trd = prices.ewm(halflife=get_halflife(long_win)).mean()
        vol_prices = prices.rolling(self.sgn_vol_win).std().fillna(method='bfill')
        
        macd = short_trd - long_trd
        q = macd / vol_prices
        trd = q / q.rolling(self.trd_vol_win).std().fillna(method='bfill')
        
        return trd

    def get_signal(self, prices):
        sgns = None

        # average multiple signals w diff time scales
        for short_win, long_win in self.trd_comb:
            sgn = calc_signal_scale(prices, short_win, long_win)

            if sgns is None:
                sgns = scale_signal(sign)
            else:
                sgns += scale_signal(sign)
        
        return sgns / len(self.trd_comb)


    @staticmethod
    def scale_signal(z):
        return z * np.exp(-z**2 / 4) / 0.89

    @staticmethod
    def get_halflife(s):
        """
        Args
            s: time scale 
        """
        return np.log(0.5) / log(1 - 1 / s)