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

def evaluate(model, test_iter, base_df, data_info, train_details, model_path=None):
    print("> Evaluate")

    if model_path is not None:
        # if model path is given load the model
        print("> Load model for inference")
        model.load_state_dict(torch.load(model_path))
        model = model.double()

    # simple strategies ----
    if model.name in ['tsmom', 'long']:
        returns = evaluate_tsmom(model=model, data=base_df, data_info=data_info)
        agg_total_returns = calc_total_returns(returns, aggregate_by='time')
        plot_total_returns(agg_total_returns)
        return 1

    model.eval()

    # evaluate test data ----
    test_loss = evaluate_model(model, test_iter, train_details)
    print(f">> Test loss: {test_loss}")

    # get predictions & calc strategy returns ----
    df_skeleton = base_df.swaplevel(axis=1)['prs']
    predictions = calc_predictions_df(model, test_iter, df_shape=df_skeleton.shape, df_index=df_skeleton.index, df_insts=df_skeleton.columns, win_step=data_info['test_win_step'])
    positions = calc_position_df(predictions, train_details['loss_type'])
    
    scaled_rts = base_df.xs('rts_scaled', axis=1, level=1, drop_level=True)
    str_returns = utils.calc_strategy_returns(positions=positions, realized_returns=scaled_rts)
    
    # save ----
    scaled_rts.to_csv("scaled_rts.csv")

    predictions_file_path = utils.get_save_path(file_label='pred', model=model.name, time_test=train_details['time_test'], file_type='csv', loss_type=train_details['loss_type'])
    predictions.to_csv(predictions_file_path)
    positions_file_path = utils.get_save_path(file_label='pos', model=model.name, time_test=train_details['time_test'], file_type='csv', loss_type=train_details['loss_type'])
    positions.to_csv(positions_file_path)
    str_returns_file_path = utils.get_save_path(file_label='rts', model=model.name, time_test=train_details['time_test'], file_type='csv', loss_type=train_details['loss_type'])
    str_returns.to_csv(str_returns_file_path)

    agg_str_total_returns = calc_total_returns(str_returns, aggregate_by='time')
    print(f">> Total strategy return from {agg_str_total_returns.index[0]} to {agg_str_total_returns.last_valid_index()}: {agg_str_total_returns[agg_str_total_returns.last_valid_index()]}")

    # plot ----
    plot_total_returns(agg_str_total_returns)

    return 1

def evaluate_model(model, data_iter, train_details):
    loss_fn = train_details['loss_fn']
    total_val_loss = 0. 
    
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            inputs = batch['inp'].double().to(device)
            labels = batch['trg'].double().to(device)

            prediction = model(inputs)

            # e.g. transformer model uses the dim: T x B x C
            if not model.batch_first:
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

def calc_predictions_df(model, data_iter, df_shape, df_index, df_insts, win_step):
    print("> Calc predictions for test data")

    predictions_df = pd.DataFrame(
        np.empty(df_shape), columns=df_insts, index=df_index)
    predictions_df[:] = np.nan
    count_df = predictions_df.copy() # tmp: to check that every cell is just touched once
    count_df[:] = 0

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            input = batch['inp'].double().to(device)
            time_id = batch['time'].numpy()
            inst = batch['inst']

            prediction = model(input)

            # dim of prediction: B/T x T/B x 1
            if not model.batch_first: 
                # time first
                prediction = prediction.permute(1,0,2).squeeze(-1).numpy() # T x B x 1 -> B x T
            else:
                # batch first
                prediction = prediction.squeeze(-1).numpy() # B x T x 1 -> B x T
                
            # insert predictions to empty df
            for sample_i in range(prediction.shape[0]):
                # the windows can overlap (determined by step parameter)..
                # .. therefore just take those predictions that don't overlapp with previous predictions
                time_id_i = time_id[sample_i] 
                # for the first window there is no previous value..
                # .. check if it's the "first" window 
                is_first_time_win = time_id_i[-1] == (input.shape[1] - 1)
                if is_first_time_win:
                    tim_id_sub_i = time_id_i[:]
                    prediction_i = prediction[sample_i, :][:]
                else:
                    tim_id_sub_i = time_id_i[-win_step:]
                    prediction_i = prediction[sample_i, :][-win_step:]

                time_i = predictions_df.index[tim_id_sub_i]

                slicer = (time_i, inst[sample_i])
                count_df.loc[slicer] += 1
                predictions_df.loc[slicer] = prediction_i

    count_df.to_csv("count_tmp.csv")

    predictions_df = predictions_df.replace(0, np.nan)
    return predictions_df


def evaluate_tsmom(model, data, data_info, do_save=True):
    str_rts = model.calc_strategy_returns(df=data)

    if do_save:
        file_path = utils.get_save_path(file_label='rts', model=model.name, time_test=data_info['time_test'])
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




    