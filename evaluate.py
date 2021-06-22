# --- --- ---
# evaluate.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import argparse
import enum
import os
import glob
import libs.utils as utils
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from runx.logx import logx
import matplotlib.pyplot as plt
from libs.position_sizing import PositionSizingHelper
from libs.losses import LossHelper, LossTypes
# data
from libs.data_loader import BaseDataLoader, DataTypes
from libs.futures_dataset import FuturesDataset
# models
from libs.models.tsmom import BasicMomentumStrategy, LongOnlyStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation Mode: Time Series Momentum with Attention-based Models')

    # saved files ----
    parser.add_argument('--model_type', type=str, nargs='?', default='ml',
                        help="Choose the model type to run", choices=['tsmom', 'long', 'ml'])

    # tsmom / long ----
    parser.add_argument('--lookback', type=int, nargs='?',
                        default=252, help="TSMOM: lookback period")

    # ml ----
    parser.add_argument('--checkpoint_path', type=str, nargs='?', default=None,
                        help="Path to the runx checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, nargs='?', default=None,
                        help="Directory to the expanding window checkpoints")
    parser.add_argument('--scaler_path', type=str, nargs='?', default=None,
                        help="TBD")

    # data ----
    parser.add_argument('--filename', type=str, nargs='?',
                        default="futures_prop.csv", help="Filename of corresponding .csv-file")
    parser.add_argument('--start_date', type=str, nargs='?',
                        default='01-01-1990', help="Start date")
    parser.add_argument('--test_date', type=str, nargs='?',
                        default='01-01-1990', help="Test date")
    parser.add_argument('--end_date', type=str, nargs='?',
                        default='01-11-2020', help="Last date")

    args = parser.parse_args()
    args.start_date = pd.to_datetime(args.start_date)
    args.test_date = pd.to_datetime(args.test_date)
    args.end_date = pd.to_datetime(args.end_date)
    return args


def main():
    args = get_args()

    if args.checkpoint_dir is not None:
        print("> Expanding window")
        checkpoints = glob.glob(args.checkpoint_dir + '/*.p')
        for i, path in enumerate(checkpoints):
            args.checkpoint_path = path
            run_test_window(args)
    elif args.checkpoint_path is not None:
        print("> Single checkpoint")
        run_test_window(args)
    elif args.model_type in ['long', 'tsmom']:
        print("> Simple strategy")
        run_test_window(args)
    else:
        raise ValueError("Valid arguments missings")

# --- --- ---
# --- --- ---


def run_test_window(args):
    print(f"\n\nEvaluate checkpoint {args.checkpoint_path}")

    # (1) Load model
    if args.model_type not in ['tsmom', 'long']:
        arch, model, train_manager = utils.load_model(
            path=args.checkpoint_path)

    # (2) Load data
    print("(2) Load data")
    index_col = 0

    if args.model_type in ['tsmom', 'long']:
        train_manager = {
            'args': {
                'filename': args.filename,
                'lead_target': 1,
                'start_date': args.start_date,
                'test_date': args.test_date,
                'end_date': args.end_date,
                'lookback': args.lookback
            }
        }

    base_loader = BaseDataLoader(
        filename=train_manager['args']['filename'], index_col=index_col,
        start_date=train_manager['args']['start_date'], end_date=train_manager['args']['end_date'],
        test_date=train_manager['args']['test_date'], lead_target=train_manager['args']['lead_target'])

    if args.model_type in ['tsmom', 'long']:
        if args.model_type == 'long':
            model = LongOnlyStrategy()
        elif args.model_type == 'tsmom':
            model = BasicMomentumStrategy(
                lookback=train_manager['args']['lookback'])

        test_dataloader = None
    elif args.model_type == 'ml':
        scaler_path = train_manager['scaler_path'] if args.scaler_path is None else args.scaler_path
        train_manager['scaler'] = utils.load_scaler(scaler_path=scaler_path)

        dataset_test = FuturesDataset(
            base_loader, DataTypes.TEST, win_size=train_manager['args']['win_len'], tau=train_manager['args']['lead_target'], step=train_manager['args']['step'], scaler=train_manager['scaler'])
        test_dataloader = DataLoader(
            dataset_test, batch_size=train_manager['args']['batch_size'], shuffle=True)

    # (3) Evaluate
    print("(3) Evaluate model")
    evaluate(model=model, data_iter=test_dataloader,
             base_df=base_loader.df[DataTypes.TEST], train_manager=train_manager)


def evaluate(model, data_iter, base_df, train_manager):
    print("> Evaluate")

    # simple strategies ----
    if model.name in ['tsmom', 'long']:
        returns = evaluate_tsmom(
            model=model, data=base_df, time_test=train_manager['args']['test_date'])
        agg_total_returns = calc_total_returns(returns, aggregate_by='time')
        return 1

    # --- ---

    # evaluate test data ----
    test_loss = evaluate_model(model, data_iter, train_manager)
    test_strategy_loss = evaluate_model_strategy(
        model, data_iter, train_manager, base_df)
    print(
        f">> Test loss: {test_loss:.6f} \t test strategy loss: {test_strategy_loss}")

    # get predictions & calc strategy returns ----
    df_skeleton = base_df.swaplevel(axis=1)['prs']
    predictions = calc_predictions_df(model, data_iter, df_shape=df_skeleton.shape,
                                      df_index=df_skeleton.index, df_insts=df_skeleton.columns,
                                      win_step=train_manager['args']['win_len'], scaler=train_manager['scaler'], loss_type=train_manager['loss_type'])
    positions = calc_position_df(predictions, train_manager['loss_type'])

    scaled_rts = base_df.xs('rts_scaled', axis=1,
                            level=1, drop_level=True)
    str_returns = utils.calc_strategy_returns(
        positions=positions, realized_returns=scaled_rts)

    # save ----
    # scaled_rts_lead.to_csv("scaled_rts_lead.csv")
    predictions_file_path = utils.get_save_path(
        file_label='pred', model=model.name, setting=train_manager['setting'], time_test=train_manager['args']['test_date'], file_type='csv')
    predictions.to_csv(predictions_file_path)
    positions_file_path = utils.get_save_path(
        file_label='pos', model=model.name, setting=train_manager['setting'], time_test=train_manager['args']['test_date'], file_type='csv')
    positions.to_csv(positions_file_path)
    str_returns_file_path = utils.get_save_path(
        file_label='rts', model=model.name, setting=train_manager['setting'], time_test=train_manager['args']['test_date'], file_type='csv')
    str_returns.to_csv(str_returns_file_path)

    # agg_str_total_returns = calc_total_returns(
    #    str_returns, aggregate_by='time')
    # print(
    #   f">> Total strategy return from {agg_str_total_returns.index[0]} to {agg_str_total_returns.last_valid_index()}: {agg_str_total_returns[agg_str_total_returns.last_valid_index()]}")

    # plot ----
    # trs_plot_path = utils.get_save_path(
    #    file_label='plot', model=model.name, setting=train_manager['setting'], time_test=train_manager['args']['test_date'], file_type='pdf')
    # plot_total_returns(agg_str_total_returns, path=trs_plot_path)

    return 1


def evaluate_model_strategy(model, data_iter, train_manager, base_df):
    df_skeleton = base_df.swaplevel(axis=1)['prs']
    scaled_rts = base_df.xs('rts_scaled', axis=1,
                            level=1, drop_level=True)

    predictions = calc_predictions_df(model, data_iter, df_shape=df_skeleton.shape,
                                      df_index=df_skeleton.index, df_insts=df_skeleton.columns,
                                      win_step=train_manager['args']['win_len'], scaler=train_manager['args']['scaler'], loss_type=train_manager['loss_type'])
    positions = calc_position_df(
        predictions, train_manager['loss_type'])
    str_returns = utils.calc_strategy_returns(
        positions=positions, realized_returns=scaled_rts, aggregate_by='time', lead=1)

    loss_fn = LossHelper.get_strategy_loss_function(
        train_manager['loss_type'])
    str_loss = loss_fn(str_returns)

    return str_loss


def evaluate_model(model, data_iter, train_manager, do_log=None):
    model.eval()

    loss_fn = train_manager['loss_fn']
    total_loss = np.zeros((len(data_iter), 2))  # batch loss & batch size

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            predictions, labels, returns = process_one_batch(
                model, batch, train_manager)

            if LossHelper.use_returns_for_loss(train_manager['loss_type']):
                loss = loss_fn(predictions, returns,
                               freq=train_manager['frequency'])
            else:
                loss = loss_fn(predictions, labels)

            batch_size = batch['inp'].shape[0]
            total_loss[i] = np.array([loss, batch_size])

    mean_loss = np.average(total_loss[:, 0], weights=total_loss[:, 1])
    return mean_loss


def calc_position_df(prediction, loss_type):
    # TBD: add more complicated position rules

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


def calc_predictions_df(model, data_iter, df_shape, df_index, df_insts, win_step, loss_type=None, scaler=None):
    predictions_df = pd.DataFrame(
        np.empty(df_shape), columns=df_insts, index=df_index)
    predictions_df[:] = np.nan
    # tmp: to check that every cell is just touched once
    count_df = predictions_df.copy()
    count_df[:] = 0

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            inst = batch['inst']
            time_id = batch['time'].cpu().numpy()
            predictions, labels, returns = process_one_batch(
                model, batch)

            # dim of prediction: B/T x T/B x 1
            if not model.batch_first:
                predictions = predictions.permute(
                    1, 0, 2).squeeze(-1).cpu().numpy()  # T x B x 1 -> B x T
            else:
                # B x T x 1 -> B x T
                predictions = predictions.squeeze(-1).cpu().numpy()

            # insert predictions to empty df
            for sample_i in range(predictions.shape[0]):
                time_id_i = time_id[sample_i]
                time_i = predictions_df.index[time_id_i]
                prediction_i = predictions[sample_i, :]
                slicer = (time_i, inst[sample_i])

                if not predictions_df.loc[slicer].isnull().all():
                    tim_id_sub_i = time_id_i[-win_step:]
                    time_i_sub = predictions_df.index[tim_id_sub_i]
                    prediction_i = prediction_i[-win_step:]
                    slicer = (time_i_sub, inst[sample_i])

                count_df.loc[slicer] += 1
                predictions_df.loc[slicer] = prediction_i

    # inverse scale trend predictions
    if (scaler is not None and scaler != "none") and LossHelper.get_prediction_type(loss_type) == 'trend':
        prediction = utils.inverse_scale_tensor(
            df=predictions_df, scaler_dict=scaler)

    # for verification
    # count_df.to_csv("count_tmp.csv")

    return predictions_df


def process_one_batch(model, batch, train_manager=None):
    inputs = batch['inp'].double().to(device)
    labels = batch['trg'].double().to(device)
    returns = batch['rts'].double().to(device)

    if model.name == 'informer':
        inputs_time_embd = batch['time_embd'].double().to(device)
        prediction, attns = model(inputs, inputs_time_embd)
    elif model.name == 'conv_momentum':
        x_static = batch['inst_id'].to(device)
        prediction = model(inputs, x_static)
    elif model.name == 'transformer':
        x_time = batch['time_embd'].double().to(device)
        x_static = batch['inst_id'].to(device)
        prediction = model(inputs, x_time, x_static)
    elif model.name == 'conv_transformer':
        x_time = batch['time_embd'].double().to(device)
        x_static = batch['inst_id'].to(device)
        prediction, _ = model(inputs, x_time, x_static)
    elif model.name == 'transformer_full':
        B, L, D = inputs.shape
        time = batch['time_embd'].double().to(device)
        entity = batch['inst_id'].to(device)
        enc = inputs[:, : -7, :]
        enc_time = time[:, : -7, :]
        enc_entity = entity

        dec_pad = torch.ones([B, 1, D]).double().to(device)
        dec_inp = inputs[:, -7:, :]
        dec = torch.cat((dec_pad, dec_inp), dim=1)
        dec_time = time[:, -7:, :]
        dec_entity = entity

        prediction = model(enc, dec, enc_time,
                           enc_entity, dec_time, dec_entity)

        # if decoder
        prediction = prediction.unsqueeze(-2)
        labels = labels[:, -1, :]
        returns = returns[:, -1, :]
    else:
        prediction = model(inputs)

    return prediction, labels, returns


def evaluate_tsmom(model, data, time_test, do_save=True):
    str_pos, str_rts = model.calc_strategy_returns(df=data)

    if do_save:
        test_time = pd.to_datetime(time_test).year
        setting = f"a-{model.name}_ty-{test_time}_"
        if model.name == "tsmom":
            setting = setting + f"lb-{model.lookback}"

        pos_path = utils.get_save_path(
            file_label='pos', model=model.name, setting=setting, time_test=time_test)
        rts_path = utils.get_save_path(
            file_label='rts', model=model.name, setting=setting, time_test=time_test)
        str_pos.to_csv(pos_path)
        str_rts.to_csv(rts_path)

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

    trs = utils.calc_total_returns_df(
        df=strategy_returns, vol_scaling=False, is_prices=False)
    trs = trs.mean(axis=axis)
    return trs


def plot_total_returns(total_returns, path=None):
    plt.plot(total_returns)
    plt.title("Total return of strategy")
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

# --- --- ---
# --- --- ---


if __name__ == "__main__":
    main()
