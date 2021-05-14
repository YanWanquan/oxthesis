# --- --- ---
# evaluate.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import os
import libs.utils as utils
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from libs.position_sizing import PositionSizingHelper
from libs.losses import LossHelper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_iter, base_df, data_info, train_details):
    print("> Evaluate")

    # simple strategies ----
    if model.name in ['tsmom', 'long']:
        returns = evaluate_tsmom(model=model, data=base_df, data_info=data_info)
        agg_total_returns = calc_total_returns(returns, aggregate_by='time')
        plot_total_returns(agg_total_returns)
        return 1

    # evaluate test data ----
    test_loss = evaluate_model(model, test_iter, train_details)
    print(f">> Test loss: {test_loss}")

    # get predictions & calc strategy returns ----
    predictions = calc_predictions_df(model, test_iter, base_df)
    positions = calc_position_df(predictions, train_details['loss_type'])
    
    scaled_rts = base_df.xs('rts_scaled', axis=1, level=1, drop_level=True)
    str_returns = utils.calc_strategy_returns(positions=positions, realized_returns=scaled_rts)
    str_returns_file_path = utils.get_results_path(file_label='rts', model=model.name, time_start=data_info['time_start'], time_end=data_info['time_end'])
    str_returns.to_csv(str_returns_file_path)

    agg_str_total_returns = calc_total_returns(str_returns, aggregate_by='time')
    print(f">> Total strategy return from {agg_str_total_returns.index[0]} to {agg_str_total_returns.last_valid_index()}: {agg_str_total_returns[agg_str_total_returns.last_valid_index()]}")

    # plot ----
    #plot_total_returns(agg_str_total_returns)

    return 1

def evaluate_model(model, data_iter, train_details):
    loss_fn = train_details['loss_fn']()
    total_val_loss = 0. 

    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            inputs = batch['inp'].double().to(device)
            labels = batch['trg'].double().to(device)

            prediction = model(inputs)

            # transformer model uses the dim: T x B x C
            if model.name == 'transformer':
                labels = labels.permute(1, 0, 2)

            total_val_loss += loss_fn(prediction, labels)
        
    return total_val_loss / (len(data_iter) - 1)


def calc_position_df(prediction, loss_type):
    # TBD: add more complicated position rules
    print("> Calc positions for test data")

    pred_type = LossHelper.get_prediction_type(loss_type)
    if pred_type == 'position':
        position = prediction
    elif pred_type == 'trend':
        position = calc_position_sizing(prediction)
    else:
        raise ValueError("Unknown prediction type!")

    return position

def calc_position_sizing(signals):
    """
    Args:
        signals (pd.DataFrame)
    """
    return PositionSizingHelper.sign_sizing_fn(signals)

def calc_predictions_df(model, data_iter, df):
    print("> Calc predictions for test data")
    model.eval()

    df = df.swaplevel(axis=1)['prs']
    predictions_df = pd.DataFrame(
        np.empty(df.shape), columns=df.columns, index=df.index)

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            input = batch['inp'].double().to(device)
            time_id = batch['time'].numpy()
            inst = batch['inst']

            prediction = model(input)
            prediction = prediction.permute(1,0,2).squeeze(-1).numpy() # B x T

            # insert predictions to empty df
            for sample_i in range(prediction.shape[0]):
                time_i = df.index[time_id[sample_i]]
                predictions_df.loc[time_i, inst[sample_i]] = prediction[sample_i, :]

    predictions_df = predictions_df.replace(0, np.nan)
    return predictions_df


def evaluate_tsmom(model, data, data_info, do_save=True):
    str_rts = model.calc_strategy_returns(df=data)

    if do_save:
        file_path = utils.get_results_path(file_label='rts', model=model.name, time_start=data_info['time_start'], time_end=data_info['time_end'])
        str_rts.to_csv(file_path)

    return str_rts

def calc_total_returns(strategy_returns, aggregate_by='time'):
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
    return trs

def plot_total_returns(total_returns):
    plt.plot(total_returns)
    plt.title("Total return of strategy")
    plt.show()




    