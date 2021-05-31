# --- --- ---
# data_loader_interface.py
# Sven Giegerich / 03.05.2021
# --- --- ---

#import futures_data_loader
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
import os
import matplotlib.pyplot as plt

from libs.models.tsmom import MACDStrategy
import libs.utils as utils

# ?! add proper factory

# ?! add proper factory
#import futures_config

# --- --- ---


class DataTypes(IntEnum):

    TRAIN = 1
    VAL = 2
    TEST = 3

    @staticmethod
    def get_string_name():
        return {
            DataTypes.TRAIN: "train",
            DataTypes.VAL: "validation",
            DataTypes.TEST: "test"
        }


# --- --- ---


""" class DataLoaderFactory:

    _data_loader_class_map = {
        'futures': futures_data_loader  # ?!
    }

    @classmethod
    def _get_valid_loaders(cls):
        return cls._data_loader_class_map().keys().sort

    @classmethod
    def _check_loader_name(cls, loader_name):
        if loader_name in cls._data_loader_class_map:
            return True
        else:
            raise ValueError(f"Unrecognized data loader {loader_name}!\n \
                Valid loaders: {','.join(_get_valid_loaders())}")

    @classmethod
    def make_data_loader(cls, loader_name):
        if cls._check_loader_name(loader_name):
            return cls._data_loader_class_map[loader_name]()

    # ?! -> add proper checks
    @classmethod
    def get_config(cls, laoder_name):
        if cls._check_loader_name(loader_name):
            return futures_configs
 """

# --- --- ---


class BaseDataLoader:

    def __init__(self, filename, index_col, start_date, end_date, test_date, lead_target):
        self.start_date = start_date
        self.end_date = end_date
        self.test_date = test_date

        self.raw_df = self.load_raw_df(filename, index_col)
        self.df = self.get_covariates(lead_target=lead_target)

        # filter afterwards (otherwise we would need to use a different threshold)
        self.raw_df = self.filter_df(
            self.raw_df, start_date=start_date, end_date=end_date)
        self.df = self.filter_df(
            self.df, start_date=start_date, end_date=end_date)

        self.raw_df = self.train_split_data(self.raw_df)
        self.df = self.train_split_data(self.df)

    def load_raw_df(self, filename, index_col, parse_dates=True, dat_first=True):
        """
        Args
            filename (str): the filename of the dataset (is joined with data folder path)
            index_col (int): index for the column informations
            parse_dates (bool): whether to parse data formates
            dat_first (bool): should the day comes first in dates?
        """
        print(f"> Load raw datset ({filename})")
        return pd.read_csv(
            os.path.join(utils.DATA_FOLDER, filename),
            index_col=0, parse_dates=parse_dates, dayfirst=dat_first
        ).sort_index(axis=0).sort_index(axis=1)

    def filter_df(self, df, start_date, end_date, col='index'):
        """
        Args
            df (pd.DataFrame)
            start_date (str): format should match the corresponding date column
            end_date (str): same
            col (str): the date column, if equal to 'index' the date column is assumed to be the index of df    
        """
        return df.query(f"{col} >= '{start_date}' and {col} <= '{end_date}'")

    def get_covariates(self, lead_target):
        smooth_window = 252
        missingness_threshold = 0.1
        vol_target = 0.15
        vol_lookback = 60
        drop_na = False

        # add covariates to the raw price data
        covariates_dict = {}
        covariates_dict['prs'] = self.get_cleaned_df(
            smooth_window=smooth_window, missingness_threshold=missingness_threshold)

        print("> Calculate covariates")

        # total returns and vol
        covariates_dict['trs'] = self.get_total_returns(
            df=covariates_dict['prs'], vol_scaling=True)
        rts = utils.calc_returns_df(
            covariates_dict['prs'], offset=1, drop_na=False)
        covariates_dict['rts_scaled'] = covariates_dict['trs'] / \
            covariates_dict['trs'].shift(1) - 1
        covariates_dict['vol_norm'] = self.get_vol_normalizer(
            df=rts, vol_target=vol_target, vol_lookback=vol_lookback)
        covariates_dict['vol'] = 1.0 / covariates_dict['vol_norm']

        covariates_dict['vol_raw'] = utils.calc_volatility_df(
            df=covariates_dict['trs'], vol_lookback=60)

        # "normalized" returns (Lim et al., 2020)
        time_scales = {'daily_rts': 1, 'monthly_rts': 20, 'quarterly_rts': 63,
                       'semianual_rts': 126, 'annual_rts': 252}
        for label, time_scale in time_scales.items():
            covariates_dict[label] = utils.calc_normalized_returns_df(
                df=covariates_dict['prs'], offset=time_scale, vol_scaling_factor=time_scale, vol_lookback=vol_lookback, drop_na=drop_na)

        # MACD signals
        trend_combinations = [(8, 24), (16, 48), (32, 96)]
        macd = MACDStrategy(trd_comb=trend_combinations)
        for short_win, long_win in trend_combinations:
            covariates_dict[f"MACD_{short_win}"] = macd.calc_signal_scale(
                prices=covariates_dict['prs'], short_win=short_win, long_win=long_win)

        # target returns: label as last column
        covariates_dict['rts_scaled_lead'] = covariates_dict['rts_scaled'].shift(
            -lead_target)  # for loss
        covariates_dict['trg'] = covariates_dict['rts_scaled'].shift(
            -lead_target)

        # concatenate all single df's to a multindex df:
        #   instruments (first level), covariates (second level), time (axis 0)
        df = pd.concat(covariates_dict.values(), axis=1,
                       keys=covariates_dict.keys()).swaplevel(axis=1)

        return df

    def train_split_data(self, df):
        data = {}
        calib_data = self.df[self.df.index < self.test_date]
        test_T = int(0.9 * len(calib_data))

        data[DataTypes.TRAIN] = calib_data.iloc[:test_T]
        data[DataTypes.VAL] = calib_data.iloc[test_T:]
        data[DataTypes.TEST] = self.df[self.df.index >= self.test_date]

        return data

    def get_cleaned_df(self, smooth_window=252, missingness_threshold=0.1):
        # forward fill missings, treat zeros as missings
        prs = self.raw_df.copy().replace(0.0, np.nan)
        prs = prs.fillna(method='ffill')

        # winsorize
        # ..

        # filter series with too many missings
        missing_ratio_insts = self.calc_missingness(
            prs, drop_na=True, missing_values=[0, np.nan])
        valid_insts = missing_ratio_insts[missing_ratio_insts <
                                          missingness_threshold].index
        prs = prs[valid_insts]
        print(
            f"> Filtered out series: {', '.join(set(missing_ratio_insts.index) - set(valid_insts))}")

        # ?! why set them np.nan when they are already np.nan (by def. of flag_inactive_sequences)?
        inactive_flags = self.flag_inactive_sequences(prs)
        prs = (prs * (1 - inactive_flags)).replace({0: np.nan})

        prs = self.winzorize_df(prs, halflife=smooth_window, threshold=3)

        return prs

    def get_total_returns(self, vol_scaling, df=None, vol_target=0.15, vol_lookback=60):
        """
        Args
            prs (pd.DataFrame): cleaned dataframe (dim: T x instruments)
        """

        if df is None:
            df = self.get_cleaned_df()
        return utils.calc_total_returns_df(df, vol_scaling=vol_scaling,
                                           vol_target=vol_target, vol_lookback=vol_lookback)

    def get_vol_normalizer(self, vol_target, vol_lookback, df=None):
        """
        Calculate the ratio (vol_target / vol) to match the volatility
        Args
            tbd
        """
        if df is None:
            df = self.get_cleaned_df()
        inactive_flags = self.flag_inactive_sequences(df)

        rts = utils.calc_returns_df(df, offset=1, drop_na=False)
        vol = utils.calc_volatility_df(
            df, ex_ante=False, vol_lookback=vol_lookback)

        normalizer = vol_target / (vol * np.sqrt(252))  # ?! replace 252
        normalizer = normalizer.fillna(method="ffill")
        normalizer = (normalizer * (1 - inactive_flags)).replace({0: np.nan})

        return normalizer

    def get_raw_df(self):
        return self.raw_df

    def get_identifiers(self):
        return list(self.raw_df.columns)

    @classmethod
    def flag_inactive_sequences(cls, df):
        """
        Flag sequences that have np.nan's at start or end
        For example, 
            [[np.nan, np.nan, 3, 4], ...] => [[True, True, False, False], ...]
            [[3, 4, np.nan, np.nan], ...] => [[False, False, True, True], ...]
        Adopted from Brian Lym.
        Args
            df (pd.DataFrame): dataframe with multiple time series (dim: T x identifier)
        """
        # Do forward pass to determine starting location
        forward_inactive_locations = cls.flag_inactive_locations(df)
        # Do backward pass to determine termination point
        reverse_inactive_locations = cls.flag_inactive_locations(
            df.sort_index(ascending=False))

        return (forward_inactive_locations + reverse_inactive_locations) == 1

    @staticmethod
    def flag_inactive_locations(df):
        """
        Flags starting sequence (till first real value) of time series that starts with np.nan.
        Adopted from Brian Lym.
        Args
            df (pd.DataFrame): dataframe with multiple time series (dim: T x identifier)
        Returns, e.g.,
            [[np.nan, np.nan, 3, 4], ...] => [[1, 1, 0, 0], ...]
            [[3, 4, np.nan, np.nan], ...] => [[0, 0, 0, 0], ...]
        """
        return (((df.isna() * -1 + 1).cumsum() == 0.0) * 1)

    @staticmethod
    def calc_missingness(df, drop_na, missing_values=[0, np.nan]):
        """
        Calculates the ratio of missing entries per time series
        Args
            df (pd.DataFrame): dataframe with multiple time series (dim: T x identifier)
            drop_na (bool): whether missings should be dropped before calculating returns
            missing_value (list): depends on missing imputation strategy
        Returns
            pd.Series: one ratio per series
        """
        return df.apply(
            lambda prs: utils.calc_returns_srs(prs, offset=1, drop_na=drop_na).isin(
                missing_values).sum() / len(prs)
        ).sort_values(ascending=True)

    @staticmethod
    def winzorize_df(df, halflife, threshold):
        """
        Winzorize a given series: cap outliers given 
            by <threshold> x expoential moving volatility
        Args
            df (pd.DataFrame): (cleaned) price series
            halflife (int): the halflife of the rolling volatility
            threshold (int): factor times the value can excced the volatility 
        """
        rolling_df = df.ewm(halflife=halflife)
        rolling_mean = rolling_df.mean()
        rolling_std = rolling_df.std()

        upper_bound = rolling_mean + threshold * rolling_std
        lower_bound = rolling_mean - threshold * rolling_std

        count_outliers = (df > upper_bound).sum() + (df < lower_bound).sum()
        count_outliers = count_outliers[count_outliers > 0].sort_values(
            ascending=False)
        print(f"> Winzorizing {list(count_outliers)} values at: \
            [{', '.join(count_outliers.index)}] (threshold: {threshold})")

        df[df > upper_bound] = upper_bound
        df[df < lower_bound] = lower_bound

        return df


# --- --- ---

# For testing purposes

if __name__ == "__main__":
    data_loader = BaseDataLoader(
        filename="futures_prop.csv", index_col=0, start_date="01/01/1990", end_date="01/01/2021", test_date="01/01/2015", lead_target=1)

    data_loader.df['DX_Close'].to_csv('test.csv')

    #prs = data_loader.get_cleaned_df()

    #total_returns = data_loader.get_total_returns()

    #vol = utils.calc_volatility_df(prs, ex_ante=True, halflife=0.5)
    #vol_norm = data_loader.get_vol_normalizer(0.15, 60)

    #print(prs[['BN_Close', 'ZP_Close', 'ZR_Close']].head())
    #print("--- ---")
    #print(total_returns[['BN_Close', 'ZP_Close', 'ZR_Close']].head())
    #print("--- ---")
    #print(vol_norm[['BN_Close', 'ZP_Close', 'ZR_Close']].head())

    # data_loader.get_data()

    # plt.plot(prs['BN_Close'])
    # plt.show()
    # plt.plot(total_returns['BN_Close'])
    # plt.show()
    # plt.plot(vol['BN_Close'])
    # plt.show()
    # plt.plot(vol_norm['BN_Close'])
    # plt.show()
