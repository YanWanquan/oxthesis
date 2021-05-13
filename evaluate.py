# --- --- ---
# evaluate.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import os
import libs.utils as utils
import pandas as pd

import matplotlib.pyplot as plt

def evaluate(model, test_data, data_info):
    print("> Evaluate")

    # calculate returns ----
    if model.name in ['tsmom', 'long']:
        str_rts = evaluate_tsmom(model=model, data=test_data, data_info=data_info)

    # total returns ----
    plot_total_returns(str_rts, aggregate_by='time')

    pass

def evaluate_tsmom(model, data, data_info, do_save=True):
    str_rts = model.calc_strategy_returns(df=data)

    if do_save:
        file_path = utils.get_results_path(file_label='rts', model=model.name, time_start=data_info['time_start'], time_end=data_info['time_end'])
        str_rts.to_csv(file_path)

    return str_rts

def plot_total_returns(strategy_returns, aggregate_by='time'):
    """
    Args:
        strategy_returns (pd.DataFrame): (dim: T x instrument)
    """
    if aggregate_by == 'instrument':
        axis = 0
    elif aggregate_by == 'time':
        axis = 1
    
    trs = utils.calc_total_returns_df(df=strategy_returns, vol_scaling=False, is_prices=False)
    trs = trs.mean(axis=axis)

    plt.plot(trs)
    plt.show()




    