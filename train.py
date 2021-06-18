# --- --- ---
# train.py
# Sven Giegerich / 03.05.2021
# --- --- ---

from math import e
import os
from operator import mod
from datetime import datetime, timedelta
from sys import platform
import argparse
import torch
from torch.nn.modules import transformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from runx.logx import logx
import pickle
from sklearn.model_selection import ParameterSampler
# utils
import libs.utils as utils
from libs.losses import LossHelper, LossTypes
from libs.models.informer import time_features
# eval
import evaluate
# data
from libs.data_loader import BaseDataLoader, DataTypes
from libs.futures_dataset import FuturesDataset
# models
from libs.models.transformer import TransformerEncoder
from libs.models.conv_transformer import ConvTransformerEncoder
from libs.models.lstm_dropout import LSTM
from libs.losses import LossTypes, LossHelper
from libs.models.informer import InformerEncoder
from libs.models.conv_momentum import ConvMomentum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_archs = {
    'transformer': TransformerEncoder,
    'lstm': LSTM,
    'conv_transformer': ConvTransformerEncoder,
    'informer': InformerEncoder,
    'conv_momentum': ConvMomentum
}

# torch.manual_seed(0)

# --- --- ---
# --- --- ---

hyper_grid = {
    'lstm': {
        'batch_size': [64, 128, 254],
        'lr': [1, 0.1, 0.01, 0.001, 0.0001],
        'max_grad_norm': [1, 0.1, 0.01, 0.001, 0.0001],
        # ----
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'd_hidden': [5, 10, 20, 40, 80],
        'n_layer': [1]
    },
    'informer': {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'max_grad_norm': [1, 0.1, 0.01, 0.001, 0.0001],
        # ---
        'attn': ['prob'],
        # 'informer_embed_type': ['fixed', 'timeF'],
        'informer_embed_type': ['simple'],
        'n_layer': [1, 2],
        'n_head': [2, 4, 8],
        'd_model': [8, 16, 32, 64],
        'd_hidden_factor': [1, 2, 4],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    'transformer': {
        'batch_size': [64, 128, 254],
        'lr': [0.01, 0.001, 0.0001],
        'max_grad_norm': [1, 0.1, 0.01, 0.001, 0.0001],
        # ---
        'd_model': [8, 16, 32, 64],
        'n_head': [1, 2, 4, 8],
        'n_layer': [1, 2, 3],
        'd_hidden_factor': [1, 2, 4],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        # embedding ----
        'embedding_add': ['projection', 'simple'],
        'embedding_pos': ['simple', 'learn'],
        'embedding_tmp': [0, 1],
        'embedding_id': [0, 1]
    },
    'conv_transformer': {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001, 0.0001],
        'max_grad_norm': [1, 0.1, 0.01, 0.001, 0.0001],
        # ---
        'd_model': [8, 16, 32, 64],
        'n_head': [1, 2, 4, 8],
        'n_layer': [1, 2, 3],
        'd_hidden_factor': [1, 2, 4],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        # ---
        'conv_len': [1, 3, 6, 9]
    },
    'conv_momentum': {
        'batch_size': [64, 128, 254],
        'lr': [0.01, 0.001, 0.0001],
        'max_grad_norm': [1, 0.1, 0.01, 0.001, 0.0001],
        # ----
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'd_hidden': [4, 12, 20, 40, 80],
        'n_layer': [1],
        # ----
        'n_head': [1, 2, 4],
        'd_hidden_factor': [1, 2, 4],
        # ----
        'informer_embed_type': ['timeF', 'momentum'],
        'attn': ['full']
    }
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Time Series Momentum with Attention-based Models')

    # choices
    loss_dict = {LossHelper.get_name(loss): loss
                 for loss in LossHelper.get_valid_losses()}
    if platform == "linux" or platform == "linux2":
        default_log_path = '/nfs/home/sveng/logs/tmp'
    elif platform == "darwin":
        default_log_path = '/Users/svengiegerich/runx/tmp'

    # runx ----
    parser.add_argument('--logdir', type=str, nargs='?',
                        default=default_log_path, help="Learning architecture")
    # main params ----
    parser.add_argument('--arch', type=str, nargs='?', choices=list(
        train_archs.keys()), default="transformer", help="Learning architecture")
    parser.add_argument('--loss_type', type=str, nargs='?', choices=list(
        loss_dict.keys()), default="sharpe", help="Loss function")
    parser.add_argument('--stopping_type', type=str, default='strategy', choices=['strategy', 'batch'],
                        help="Either evaluate and early stop by batch loss or strategy loss")
    parser.add_argument('--random_search_len', type=int, default=None,
                        help="Run the hyperparam inside the expanding window? Also specifiy --logdir")
    # data ----
    parser.add_argument('--filename', type=str, nargs='?',
                        default="futures_prop.csv", help="Filename of corresponding .csv-file")
    parser.add_argument('--frequency', type=str, nargs='?',
                        default='d', choices=['d'], help="Frequency of the observations")
    parser.add_argument('--start_date', type=str, nargs='?',
                        default="01/01/1990", help="Start date")
    parser.add_argument('--test_date', type=str, nargs='?',
                        default="01/01/2000", help="Test date")
    parser.add_argument('--test_win', type=int, nargs='?',
                        default=None, help="If set, it runs a expanding window approach; expects the window length in years")
    parser.add_argument('--end_date', type=str, nargs='?',
                        default="10/01/2005", help="Last date")
    parser.add_argument('--scaler', type=str, nargs='?', choices=['none', 'minmax', 'standard'],
                        default="none", help="Sklearn scaler to use")
    # window ----
    parser.add_argument('--lead_target', type=int, nargs='?',
                        default=1, help="The #lead between input and target")
    parser.add_argument('--win_len', type=int, nargs='?',
                        default=63, help="Window length to slice data")
    parser.add_argument('--step', type=int, nargs='?',
                        default=63, help="If step is not equal win_len the windows will overlap")
    # training ----
    parser.add_argument('--epochs', type=int, nargs='?',
                        default=10, help="Number of maximal epochs")
    parser.add_argument('--patience', type=int, nargs='?',
                        default=25, help="Early stopping rule")
    parser.add_argument('--lr', type=float, nargs='?',
                        default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, nargs='?',
                        default=128, help="Batch size for training")
    parser.add_argument('--max_grad_norm', type=float, nargs='?',
                        default=0.5, help="Max gradient norm for clipping")
    parser.add_argument('--dropout', type=float, nargs='?',
                        default=0.1, help="Dropout rate applied to all layers of an arch")
    # model specific params
    # .. all models
    parser.add_argument('--n_layer', type=int, nargs='?',
                        default=1, help="Number of sub-encoder layers in transformer")
    parser.add_argument('--d_hidden', type=int, nargs='?',
                        default=12, help="Dimension of feedforward network model (transformer) or in hidden state h (lstm)")
    # .. transformer
    parser.add_argument('--d_model', type=int, nargs='?',
                        default=20, help="Number of features in the encoder inputs")
    parser.add_argument('--n_head', type=int, nargs='?',
                        default=4, help="Number of heads in multiheadattention models")
    parser.add_argument('--d_hidden_factor', type=int, nargs='?',
                        default=0, help="d_dim = d_hidden_factor * d_hidden")
    # .. convolutional transformer
    parser.add_argument('--conv_len', type=int, nargs='?',
                        default=1, help="ConvTransformer: kernel size for query-key pair")
    # .. informer
    parser.add_argument('--attn', type=str, nargs='?', choices=['prob', 'full'],
                        default='prob', help="Informer: choose attention mechanism")
    parser.add_argument('--factor', type=int, nargs='?',
                        default=5, help="Informer: sampling factor for prob attn")
    parser.add_argument('--informer_embed_type', type=str, nargs='?', choices=['fixed', 'timeF', 'simple', 'momentum'],
                        default='fixed', help="Informer: choose embedding layer")
    # .. embedding
    parser.add_argument('--embedding_add', type=str, nargs='?', choices=['projection', 'simple'],
                        default='projection', help="tbd")
    parser.add_argument('--embedding_pos', type=str, nargs='?', choices=['simple', 'learn'],
                        default='simple', help="tbd")
    parser.add_argument('--embedding_tmp', type=int, nargs='?', choices=[0, 1],
                        default=1, help="tbd")
    parser.add_argument('--embedding_id', type=int, nargs='?', choices=[0, 1],
                        default=1, help="tbd")

    args = parser.parse_args()
    args.embedding_tmp = bool(args.embedding_tmp)
    args.embedding_id = bool(args.embedding_id)
    return args


def main():
    args = get_args()

    args.start_date = pd.to_datetime(args.start_date)
    args.end_date = pd.to_datetime(args.end_date)

    if args.test_win is not None:
        print("> Start expanding window training")
        args.test_date = pd.to_datetime(
            args.test_date)  # + pd.offsets.DateOffset(years=args.test_win)
        args.stop_date = pd.to_datetime(args.end_date)
        args.end_date = pd.to_datetime(
            args.test_date) + pd.offsets.DateOffset(years=args.test_win)
        args.do_log = False  # runx works just for one model per experiment

        best_val_loss = {}
        best_test_loss = {}
        best_settings = {}
        log_val_loss = {}
        log_test_loss = {}
        checkpoints = {}
        wins_len = []
        while args.test_date < args.stop_date:

            if args.random_search_len:
                args_dict_raw = vars(args)
                checkpoints[args.test_date] = []

                # sample parameters randomly ----
                param_list = list(ParameterSampler(
                    hyper_grid[args.arch], n_iter=args.random_search_len))

                # run random search ----
                search_best_score = np.Inf
                for i in range(args.random_search_len):

                    # update args ----
                    args_dict_i = args_dict_raw
                    params = param_list[i]
                    for param, value in params.items():
                        args_dict_i[param] = value
                    args_dict_i = utils.DotDict(args_dict_i)

                    # run window ----
                    val_loss_i, test_loss_i, setting_i, checkpoint_i = run_training_window(
                        args_dict_i)

                    # logs
                    checkpoints[args.test_date].append(checkpoint_i)
                    log_val_loss[setting_i] = val_loss_i
                    log_test_loss[setting_i] = test_loss_i

                    if val_loss_i < search_best_score:
                        print(
                            f"\n> Found better hyperparams ({search_best_score:.6f} --> {val_loss_i:.6f}): {setting_i}")
                        search_best_score = val_loss_i
                        search_best_checkpoint = checkpoint_i
                        best_val_loss[args.test_date], best_test_loss[args.test_date], best_settings[args.test_date] = (
                            val_loss_i, test_loss_i, setting_i)

                # clean all checkpoints except best one
                for file in checkpoints[args.test_date]:
                    if file != search_best_checkpoint:
                        os.remove(file)

            else:
                best_val_loss[args.test_date], best_test_loss[args.test_date], best_settings[args.test_date], _ = run_training_window(
                    args)

            # update expanding window
            wins_len.append((args.end_date - args.test_date) /
                            timedelta(days=365))
            args.test_date = pd.to_datetime(
                args.test_date) + pd.offsets.DateOffset(years=args.test_win)
            args.end_date = pd.to_datetime(
                args.test_date) + pd.offsets.DateOffset(years=args.test_win)
            if args.end_date > args.stop_date:
                args.end_date = args.stop_date

        # log settings
        if args.random_search_len:
            df_log = pd.concat([
                pd.Series(log_val_loss.values(),
                          log_val_loss.keys(), name="val_loss"),
                pd.Series(log_test_loss.values(),
                          log_test_loss.keys(), name="test_loss"),
            ], axis=1).reset_index()
            df_log = df_log.rename(columns={'index': 'setting'})
            df_log.index.name = 'id'
            df_log.to_csv(
                args.logdir + "/exp-win_random_log_hyper.csv")

        # best settings per window
        best_val_mean = np.average(
            list(best_val_loss.values()), weights=wins_len)
        best_test_mean = np.average(
            list(best_test_loss.values()), weights=wins_len)
        df_results = pd.DataFrame({
            'val_loss': best_val_loss,
            'test_loss': best_test_loss,
            'best_settings': best_settings
        })
        df_results.index.name = "window"
        df_results.to_csv(
            args.logdir + f"/exp-win_random_arch-{args.arch}_-vl-{best_val_mean:.6f}_tl-{best_test_mean:.6f}.csv")

        print("\nEnd of expanding window training")
        print(
            f"> Finished expanding window training \t val loss: {best_val_mean}")
        print(
            f"> Finished expanding window training \t test loss: {best_test_mean}")
    elif args.test_date is not None:
        print("> Start single window training")
        args.do_log = True

        val_loss, test_loss, setting, checkpoint = run_training_window(args)
    else:
        raise ValueError("Either test_win_len or test_date needs to be set!")

# --- --- ---
# --- --- ---


def run_training_window(args):
    print(f"\n> Train with test date {args.test_date}")

    if args.logdir is not None and args.do_log:
        args_log = args  # runx cann't deal with pd.DateTime objects
        args_log.start_date = str(args_log.start_date)
        args_log.test_date = str(args_log.test_date)
        args_log.end_date = str(args_log.end_date)
        logx.initialize(logdir=args.logdir, coolname=True,
                        tensorboard=True, hparams=vars(args_log))

    # (1) load data ----
    print("(1) Load data")
    index_col = 0

    base_loader = BaseDataLoader(
        filename=args.filename, index_col=index_col, start_date=args.start_date, end_date=args.end_date, test_date=args.test_date, lead_target=args.lead_target)

    dataset_train = FuturesDataset(
        base_loader, DataTypes.TRAIN, win_size=args.win_len, tau=args.lead_target, step=args.step, scaler_type=args.scaler)
    # save scaler for inverse transformation at evaluation
    scaler_path = utils.save_scaler(dataset_train.scaler, args.filename,
                                    args.start_date, args.test_date)
    dataset_val = FuturesDataset(
        base_loader, DataTypes.VAL, win_size=args.win_len, tau=args.lead_target, step=args.step, scaler_type=args.scaler)
    dataset_test = FuturesDataset(
        base_loader, DataTypes.TEST, win_size=args.win_len, tau=args.lead_target, step=args.step, scaler_type=args.scaler)

    train_dataloader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True)

    msg_sample_sizes = f"> Train sample size: {len(train_dataloader) * args.batch_size} \t val sample size: {len(val_dataloader) * args.batch_size}  \t test sample size: {len(test_dataloader) * args.batch_size}"
    if args.do_log:
        logx.msg(msg_sample_sizes)
    else:
        print(msg_sample_sizes)

    # (2) define training meta-data
    print("(2) Setup training manager")
    loss_type = LossHelper.get_loss_type(args.loss_type)
    train_manager = {
        # args
        'args': args if type(args) == utils.DotDict else vars(args),
        # loss
        'loss_label': args.loss_type,
        'loss_type': loss_type,
        'loss_fn': LossHelper.get_loss_function(loss_type),
        # learning
        'lr': args.lr,
        'patience': args.patience,
        'epochs': args.epochs,
        # data
        'frequency': args.frequency,
        'year_test': pd.to_datetime(args.test_date).year,
        # scaler
        'scaler_path': scaler_path
    }

    # (3) build model ----
    print(f"(3) Build model: {args.arch}")
    d_input = len(dataset_train.INP_COLS)
    d_output = 1

    if args.arch == 'transformer':
        if args.d_hidden_factor > 0:
            args.d_hidden = args.d_hidden_factor * args.d_model

        n_categories = len(dataset_train.inst_lookup.keys())

        model = TransformerEncoder(
            d_model=args.d_model, d_input=d_input, d_output=d_output, n_head=args.n_head,
            n_layer=args.n_layer, d_hidden=args.d_hidden, dropout=args.dropout, win_len=args.win_len,
            embedding_add=args.embedding_add, embedding_pos=args.embedding_pos, embedding_tmp=args.embedding_tmp,
            embedding_entity=args.embedding_id, n_categories=n_categories, loss_type=train_manager['loss_type'])
    elif args.arch == 'conv_transformer':
        if args.d_hidden_factor > 0:
            args.d_hidden = args.d_hidden_factor * args.d_model

        do_sparse = False
        do_scale_att = False

        # args needed: sparse (bool), embd_drop (float), attn_pdrop (float), resid_pdrop (float), scale_att (bool), q_len (int), sub_len (int)
        args_conv_transf = {
            # convolution
            'q_len': args.conv_len,  # kernel size for generating key-query
            # sparse attention
            'sparse': do_sparse,
            'sub_len': 1,  # sub_len of sparse attention
            # dropout
            'embd_pdrop': args.dropout,
            'attn_pdrop': args.dropout,
            'resid_pdrop': args.dropout,
            # others
            'scale_att': do_scale_att,
        }

        model = ConvTransformerEncoder(args=args_conv_transf, d_input=d_input, n_head=args.n_head, n_layer=args.n_layer,
                                       d_model=args.d_model, win_len=args.win_len, d_output=d_output, loss_type=train_manager['loss_type'])
    elif args.arch == 'lstm':
        dropout = 0.
        dropoutw = args.dropout
        dropouti = args.dropout
        dropouto = args.dropout
        args.d_hidden = [args.d_hidden for l in range(args.n_layer)]
        model = LSTM(d_input=d_input, d_output=d_output,
                     d_hidden=args.d_hidden, n_layer=args.n_layer, dropout=dropout, dropouti=dropouti, dropoutw=dropoutw, dropouto=dropouto, loss_type=train_manager['loss_type'])
        args.d_hidden = args.d_hidden[0]
    elif args.arch == 'informer':
        if args.d_hidden_factor > 0:
            args.d_hidden = args.d_hidden_factor * args.d_model

        # tmp: at the moment no hyperparamter
        freq = 'd'  # daily
        factor = args.factor  # factor to sample for prob attention
        d_ff = args.d_model
        attn = args.attn  # 'full' or 'prob'
        embed_type = args.informer_embed_type  # could be changed to learnable
        # if n_layer > 1: each succ layer will be reduced by 2
        do_distil = False
        output_attention = True
        win_len = args.win_len

        if attn == 'full':
            print("> Sampling factor not effective in combination with full attention")

        if args.arch == 'informer':
            model = InformerEncoder(enc_in=d_input, c_out=d_output, factor=factor,
                                    loss_type=train_manager['loss_type'], d_model=args.d_model, n_heads=args.n_head,
                                    e_layers=args.n_layer, d_ff=d_ff, dropout=args.dropout, attn=attn, embed_type=embed_type,
                                    freq=freq, output_attention=output_attention, distil=do_distil, win_len=win_len)
        elif args.arch == 'informer_full':
            raise NotImplementedError()
            dec_in = 63 + args.step
            c_out = d_output  # ?!
            out_len = args.pred_len
            d_layers = 1
            do_distil = True

            model = Informer(loss_type=train_manager['loss_type'], enc_in=d_input, dec_in=dec_in, c_out=c_out, out_len=out_len, factor=factor, d_model=args.d_model, n_heads=args.n_head,
                             e_layers=args.n_layer, d_layers=d_layers, d_ff=d_ff, dropout=args.dropout, attn=attn, embed=embed_type,
                             freq=freq, output_attention=output_attention, distil=do_distil
                             )

            raise NotImplementedError("Need to code input first!")
        else:
            raise ValueError()
    elif args.arch == 'conv_momentum':
        # for static
        use_embed = 1
        n_categories = len(dataset_train.inst_lookup.keys())
        embed_type = args.informer_embed_type

        dropout = 0.
        dropoutw = args.dropout
        dropouti = args.dropout
        dropouto = args.dropout
        args.d_hidden = [args.d_hidden for l in range(args.n_layer)]

        d_model = args.d_hidden[-1]
        len_input_window = args.win_len

        if args.d_hidden_factor > 0:
            args.d_attn_hidden = args.d_hidden_factor * args.d_model
        else:
            args.d_attn_hidden = args.d_model

        model = ConvMomentum(
            d_input=d_input, d_output=d_output, d_hidden=args.d_hidden, n_layer_lstm=args.n_layer,
            len_input_window=len_input_window, n_head=args.n_head, d_model=d_model, d_attn_hidden=args.d_attn_hidden, n_layer_attn=1,
            n_categories=n_categories, use_embed=use_embed,
            dropout=dropout, dropouti=dropouti, dropoutw=dropoutw, dropouto=dropouto, loss_type=train_manager['loss_type'])
        args.d_hidden = args.d_hidden[0]
    else:
        raise NotImplementedError("Architecture not implemented yet.")

    # (4) train model ----
    print("(4) Start training")

    # label the experiment
    train_manager['setting'] = 'a-{}_l-{}_ty-{}_bs-{}_lr-{}_pa-{}_gn-{}_wl-{}_ws-{}_nl-{}_dh-{}_dr-{}'.format(args.arch, args.loss_type, pd.to_datetime(
        args.test_date).year, args.batch_size, args.lr, args.patience, args.max_grad_norm, args.win_len, args.step, args.n_layer, args.d_hidden, args.dropout)
    if model.name in ['transformer', 'conv_transformer', 'informer', 'conv_momentum']:
        train_manager['setting'] = train_manager['setting'] + \
            '_dm-{}_nh-{}'.format(args.d_model, args.n_head)
        # plus embedding
        train_manager['setting'] = train_manager['setting'] + \
            '_embAdd-{}_embPos-{}_embT-{}_embID-{}'.format(
                args.embedding_add, args.embedding_pos, args.embedding_tmp, args.embedding_id)
    if model.name == 'conv_transformer':
        train_manager['setting'] = train_manager['setting'] + \
            '_ql-{}'.format(args_conv_transf['q_len'])
    if model.name in ['informer', 'conv_momentum']:
        train_manager['setting'] = train_manager['setting'] + \
            '_attn-{}_informerEmbed-{}_factor-{}'.format(
                args.attn, args.informer_embed_type, args.factor)
    if args.do_log:
        logx.msg(f"Setting: {train_manager['setting']}")
    else:
        print(f"Setting: {train_manager['setting']}")

    # tmp: informer notebook
    # pickle.dump({
    #     'train_df': base_loader.df[DataTypes.TRAIN],
    #     'val_df': base_loader.df[DataTypes.VAL],
    #     'test_df': base_loader.df[DataTypes.TEST]
    # }, open("data_df.p", 'wb'))
    # exit()

    best_checkpoint_path, val_loss = train(model=model, train_iter=train_dataloader,
                                           val_iter=val_dataloader, train_manager=train_manager,
                                           do_log=args.do_log, val_df=base_loader.df[DataTypes.VAL])
    print("--- --- ---")

    # (5) test model ----
    _, model, train_manager = utils.load_model(path=best_checkpoint_path)

    if train_manager['args']['stopping_type'] == 'strategy':
        test_loss = evaluate_iter(model=model, data_iter=test_dataloader,
                                  train_manager=train_manager, do_log=False, do_strategy=True, base_df=base_loader.df[DataTypes.TEST])
    else:
        test_loss = evaluate_iter(model=model, data_iter=test_dataloader,
                                  train_manager=train_manager, do_log=False, do_strategy=False)

    print(f">> Val loss: {val_loss:.6f}")
    print(f">> Test loss: {test_loss:.6f}")

    print("(6) Finished training")
    return (val_loss, test_loss, train_manager['setting'], best_checkpoint_path)


def train(model, train_iter, val_iter, train_manager, do_log=False, val_df=None):
    model = model.to(device).double()
    best_val_score = np.inf

    # train manager ----
    train_manager['optimizer'] = torch.optim.Adam(
        model.parameters(), lr=train_manager['lr'])

    stopping_path = train_manager['args']['logdir'] + "/" + "opt_" + \
        train_manager['setting'] + '.p'
    early_stopping = utils.EarlyStopping(
        patience=train_manager['patience'], path=stopping_path, verbose=True)

    # run training ----
    for epoch_i in range(train_manager['epochs']):
        epoch_loss = run_epoch(
            model=model, train_iter=train_iter, train_manager=train_manager, epoch_i=epoch_i, do_log=do_log)

        val_loss = evaluate_iter(
            model=model, data_iter=val_iter, train_manager=train_manager)
        val_str_loss = evaluate_iter(
            model=model, data_iter=val_iter, train_manager=train_manager, do_strategy=True, base_df=val_df)

        if val_loss < best_val_score:
            best_val_score = val_loss

        # verb ----
        epoch_print = f">> Train Epoch {epoch_i + 1}\t -- avg --\t train batch loss: {epoch_loss:.6f}\t val batch loss: {val_loss:.6f} \t  val strategy loss: {val_str_loss:.6f}"
        if do_log:
            logx.msg(epoch_print)
            logx.add_scalar("Loss/val", val_loss, epoch_i)

            metrics_train = {'loss': epoch_loss}
            metrics_val = {'loss': val_loss}
            # to be extented
            logx.metric(phase='train', metrics=metrics_train,
                        epoch=epoch_i + 1)
            logx.metric(phase='val', metrics=metrics_val,
                        epoch=epoch_i + 1)
        else:
            print(epoch_print)

        # save ----
        save_dict = {'epoch': epoch_i + 1,
                     'arch': model.name,
                     'train_manager': train_manager,
                     'model': model,
                     'optimizer': train_manager['optimizer'].state_dict()}

        # save the current epoch & overwrite old checkpoint (filename: <setting>.tmp)
        checkpoint_path = train_manager['args']['logdir'] + \
            "/" + train_manager['setting'] + ".tmp"
        pickle.dump(save_dict, open(checkpoint_path, 'wb'))

        # early stopping ----
        if train_manager['args']['stopping_type'] == 'strategy':
            checkpoint_loss = val_str_loss
        else:
            checkpoint_loss = val_loss

        early_stopping(checkpoint_loss, model)

        if early_stopping.do_save_model:
            early_stopping.save_checkpoint(
                val_loss=checkpoint_loss, dict=save_dict)
        if early_stopping.early_stop:
            print(f"> Early stopping")
            break

    best_val_loss = -early_stopping.best_score
    return (early_stopping.path, best_val_loss)


def run_epoch(model, train_iter, train_manager, epoch_i=None, do_log=False):
    model.train()

    optimizer = train_manager['optimizer']
    loss_fn = train_manager['loss_fn']
    max_grad_norm = train_manager['args']['max_grad_norm']

    loss_epoch = np.zeros((len(train_iter), 2))  # batch loss & batch size

    # tmp: Zihao
    # stack_input_numpy = []
    # stack_labels_numpy = []

    for i, batch in enumerate(train_iter):
        inputs = batch['inp'].double().to(device)
        labels = batch['trg'].double().to(device)
        returns = batch['rts'].double().to(device)

        # tmp: Zihao
        # stack_input_numpy.append(inputs.cpu().numpy())
        # stack_labels_numpy.append(labels.cpu().numpy())
        # continue

        optimizer.zero_grad()

        # time embedding?
        if model.name == 'informer':
            inputs_time_embd = batch['time_embd'].double().to(device)
            prediction, attns = model(inputs, inputs_time_embd)
        elif model.name == 'conv_momentum':
            inputs_time_embd = batch['time_embd'].double().to(device)
            x_static = batch['inst_id'].to(device)
            prediction = model(inputs, inputs_time_embd, x_static)
        elif model.name == 'transformer':
            x_time = batch['time_embd'].double().to(device)
            x_static = batch['inst_id'].to(device)
            prediction = model(inputs, x_time, x_static)
        else:
            prediction = model(inputs)

        if LossHelper.use_returns_for_loss(train_manager['loss_type']):
            loss = loss_fn(prediction, returns,
                           freq=train_manager['frequency'])
        else:
            loss = loss_fn(prediction, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = batch['inp'].shape[0]
        loss_epoch[i] = np.array([loss, batch_size])

        # log results
        if epoch_i is not None and i % 5 == 0:
            if do_log:
                writer_path = f"Loss/train/{train_manager['loss_label']}/{train_manager['year_test']}"
                logx.add_scalar(writer_path, loss, epoch_i *
                                len(train_iter) + i)
            print_msg = f">> Train Epoch {epoch_i+1}\t batch {i}\t train batch loss: {loss:.6f}"
            if do_log:
                logx.msg(print_msg)
            else:
                print(print_msg)

    # tmp: Zihao
    # input_numpy = np.stack(stack_input_numpy[0:-1], axis=0)
    # labels_numpy = np.stack(stack_labels_numpy[0:-1], axis=0)
    # print(input_numpy.shape)
    # pickle.dump(input_numpy, open('input_numpy.p', 'wb'))
    # pickle.dump(labels_numpy, open('labels_numpy.p', 'wb'))
    # exit()

    mean_loss_epoch = np.average(loss_epoch[:, 0], weights=loss_epoch[:, 1])
    return mean_loss_epoch


def evaluate_iter(model, data_iter, train_manager, do_strategy=False, base_df=None, do_log=False):
    if do_strategy and base_df is not None:
        # strategy loss ----
        df_skeleton = base_df.swaplevel(axis=1)['prs']
        scaled_rts = base_df.xs('rts_scaled', axis=1,
                                level=1, drop_level=True)

        predictions = evaluate.calc_predictions_df(model, data_iter, df_shape=df_skeleton.shape,
                                                   df_index=df_skeleton.index, df_insts=df_skeleton.columns,
                                                   win_step=train_manager['args']['win_len'], scaler=train_manager['args']['scaler'], loss_type=train_manager['loss_type'])
        positions = evaluate.calc_position_df(
            predictions, train_manager['loss_type'])
        str_returns = utils.calc_strategy_returns(
            positions=positions, realized_returns=scaled_rts, aggregate_by='time', lead=1)

        loss_fn = LossHelper.get_strategy_loss_function(
            train_manager['loss_type'])
        str_loss = loss_fn(str_returns)

        return str_loss
    else:
        # batch loss ----
        return evaluate.evaluate_model(model, data_iter, train_manager, do_log=do_log)

# --- --- ---
# --- --- ---


if __name__ == "__main__":
    main()
