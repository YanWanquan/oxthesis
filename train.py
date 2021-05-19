# --- --- ---
# train.py
# Sven Giegerich / 03.05.2021
# --- --- ---

from datetime import datetime
from sys import platform
import argparse
import torch
from torch.nn.modules import transformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from runx.logx import logx
# utils
import libs.utils as utils
from libs.losses import LossHelper, LossTypes
from libs.models.informer import time_features
# eval
from evaluate import evaluate_model
# data
from libs.data_loader import BaseDataLoader, DataTypes
from libs.futures_dataset import FuturesDataset
# models
from libs.models.transformer import TransformerEncoder
from libs.models.mlp import MLP
from libs.models.conv_transformer import ConvTransformerEncoder
from libs.models.lstm import LSTM
from libs.losses import LossTypes, LossHelper
from libs.models.informer import InformerEncoder

# To-Do
# - add the option not to use logx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_archs = {
    'transformer': TransformerEncoder,
    'lstm': LSTM,
    'conv_transformer': ConvTransformerEncoder,
    'informer': InformerEncoder
}

# --- --- ---
# --- --- ---


def get_args():
    parser = argparse.ArgumentParser(
        description='Time Series Momentum with Attention-based Models')

    # choices
    loss_dict = {LossHelper.get_name(loss): loss
                 for loss in LossHelper.get_valid_losses()}
    if platform == "linux" or platform == "linux2":
        default_log_path = '/nfs/home/sveng/runx/tmp'
    elif platform == "darwin":
        default_log_path = '/Users/svengiegerich/runx/tmp'

    # runx ----
    parser.add_argument('--logdir', type=str, nargs='?',
                        default=default_log_path, help="Learning architecture")
    # main params ----
    parser.add_argument('--arch', type=str, nargs='?', choices=list(
        train_archs.keys()), default="transformer", help="Learning architecture")
    parser.add_argument('--loss_type', type=str, nargs='?', choices=list(
        loss_dict.keys()), default="mse", help="Loss function")
    # data ----
    parser.add_argument('--filename', type=str, nargs='?',
                        default="futures_prop.csv", help="Filename of corresponding .csv-file")
    parser.add_argument('--start_date', type=str, nargs='?',
                        default="01/01/1990", help="Start date")
    parser.add_argument('--test_date', type=str, nargs='?',
                        default="01/01/1995", help="Test date")
    parser.add_argument('--end_date', type=str, nargs='?',
                        default="01/01/2020", help="Last date")
    # window ----
    parser.add_argument('--lead_target', type=int, nargs='?',
                        default=1, help="The #lead between input and target")
    parser.add_argument('--win_len', type=int, nargs='?',
                        default=63, help="Window length to slice data")
    parser.add_argument('--step', type=int, nargs='?',
                        default=20, help="If step is not equal win_len the windows will overlap")
    # training ----
    parser.add_argument('--epochs', type=int, nargs='?',
                        default=10, help="Number of maximal epochs")
    parser.add_argument('--patience', type=int, nargs='?',
                        default=25, help="Early stopping rule")
    parser.add_argument('--lr', type=float, nargs='?',
                        default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, nargs='?',
                        default=32, help="Batch size for training")
    parser.add_argument('--dropout', type=float, nargs='?',
                        default=0.1, help="Dropout rate applied to all layers of an arch")
    # model specific params
    # .. all models
    parser.add_argument('--n_layer', type=int, nargs='?',
                        default=1, help="Number of sub-encoder layers in transformer")
    parser.add_argument('--d_hidden', type=int, nargs='?',
                        default=20, help="Dimension of feedforward network model")
    # .. transformer
    parser.add_argument('--d_model', type=int, nargs='?',
                        default=60, help="Number of features in the encoder inputs")
    parser.add_argument('--n_head', type=int, nargs='?',
                        default=6, help="Number of heads in multiheadattention models")
    parser.add_argument('--d_hidden_factor', type=int, nargs='?',
                        default=0, help="d_dim = d_hidden_factor * d_hidden")
    # .. convolutional transformer
    parser.add_argument('--conv_len', type=int, nargs='?',
                        default=1, help="ConvTransformer: kernel size for query-key pair")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.logdir is not None:
        do_log = True
        logx.initialize(logdir=args.logdir, coolname=True, tensorboard=True)
    else:
        do_log = False

    # (1) load data ----
    print("(1) Load data")
    index_col = 0
    train_batch_size = args.batch_size
    val_batch_size = 254
    test_batch_size = 254

    base_loader = BaseDataLoader(
        filename=args.filename, index_col=index_col, start_date=args.start_date, end_date=args.end_date, test_date=args.test_date, lead_target=args.lead_target)

    dataset_train = FuturesDataset(
        base_loader, DataTypes.TRAIN, win_size=args.win_len, tau=args.lead_target, step=args.step)
    dataset_val = FuturesDataset(
        base_loader, DataTypes.VAL, win_size=args.win_len, tau=args.lead_target, step=args.step)
    dataset_test = FuturesDataset(
        base_loader, DataTypes.TEST, win_size=args.win_len, tau=args.lead_target, step=args.step)
    train_dataloader = DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(
        dataset_val, batch_size=val_batch_size, shuffle=False)
    test_dataloader = DataLoader(
        dataset_test, batch_size=test_batch_size, shuffle=False)

    # (2) define training meta-data
    print("(2) Setup training manager")
    loss_type = LossHelper.get_loss_type(args.loss_type)
    train_manager = {
        # loss
        'loss_label': args.loss_type,
        'loss_type': loss_type,
        'loss_fn': LossHelper.get_loss_function(loss_type),
        # learning
        'lr': args.lr,
        'patience': args.patience,
        'epochs': args.epochs,
        # data
        'year_test': pd.to_datetime(args.test_date).year
    }

    # (3) build model ----
    print(f"(3) Build model: {args.arch}")
    d_input = len(dataset_train.INP_COLS)
    d_output = 1

    if args.d_hidden_factor > 0:
        args.d_hidden = args.d_hidden_factor * args.d_model

    if args.arch == 'transformer':
        len_input_window = args.win_len
        len_output_window = len_input_window

        # check args
        if args.d_model % args.n_head != 0:
            # the dimensionality of multihead attention needs to be divisible by dim of encoder input
            raise ValueError(
                "Need to change arg n_head: arg d_model needs to be divisible by arg n_head")

        model = TransformerEncoder(d_model=args.d_model, d_input=d_input, d_output=d_output, n_head=args.n_head, n_layer=args.n_layer, d_hidden=args.d_hidden,
                                   dropout=args.dropout, len_input_window=len_input_window, len_output_window=len_output_window, loss_type=train_manager['loss_type'], device=device)
    elif args.arch == 'conv_transformer':
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
        model = LSTM(d_input=d_input, d_output=d_output,
                     d_hidden=args.d_hidden, n_layer=args.n_layer, loss_type=train_manager['loss_type'])
    elif args.arch == 'informer':
        # tmp: at the moment no hyperparamter
        freq = 'd'  # daily
        factor = 5  # factor to sample for prob attention
        d_ff = args.d_model
        attn = 'prob'
        embed = 'fixed'  # could be changed to learnable
        do_distil = True
        output_attention = False
        model = InformerEncoder(enc_in=d_input, c_out=d_output, factor=factor, d_model=args.d_model, n_heads=args.n_head,
                                e_layers=args.n_layer, d_ff=d_ff, dropout=args.dropout, attn=attn, embed=embed, freq=freq, output_attention=output_attention, distil=do_distil)
    else:
        raise NotImplementedError("Architecture not implemented yet.")

    # (4) train model ----
    print("(4) Start training")

    if do_log:
        setting = '{}_{}_ty-{}_wl-{}_ws-{}_nl-{}_dh-{}'.format(args.arch, args.loss_type, pd.to_datetime(
            args.test_date).year, args.win_len, args.step, args.n_layer, args.d_hidden)
        if model.name in ['transformer', 'conv_transformer', 'informer']:
            setting = setting + '_nh-{}'.format(args.n_head,)
        if model.name == 'conv_transformer':
            setting = setting + '_ql-{}'.format(args_conv_transf['q_len'])
        logx.msg(f"Setting: {setting}")

    train(model=model, train_iter=train_dataloader,
          val_iter=val_dataloader, train_manager=train_manager, do_log=do_log)

    print("(5) Finished round")

# --- --- ---
# --- --- ---


def train(model, train_iter, val_iter, train_manager, do_log=False):
    model = model.to(device).double()
    best_val_score = np.inf

    # train manager ----
    train_manager['optimizer'] = torch.optim.AdamW(
        model.parameters(), lr=train_manager['lr'])

    # run training ----
    for epoch_i in range(train_manager['epochs']):
        epoch_loss = run_epoch(
            model=model, train_iter=train_iter, train_manager=train_manager, epoch_i=epoch_i, do_log=do_log)
        val_loss = evaluate_validation(
            model=model, val_iter=val_iter, train_manager=train_manager)

        if val_loss < best_val_score:
            best_val_score = val_loss

        # verb ----
        epoch_print = f">> Epoch Step: {epoch_i + 1}\t val loss: {val_loss}\t train loss: {epoch_loss}"
        if do_log:
            logx.msg(epoch_print)
            logx.add_scalar("Loss/val", val_loss, epoch_i)

            metrics_train = {'loss': epoch_loss.data.cpu().numpy()}
            metrics_val = {'loss': val_loss.data.cpu().numpy()}
            logx.metric(phase='train', metrics=metrics_train,
                        epoch=epoch_i + 1)
            logx.metric(phase='val', metrics=metrics_val,
                        epoch=epoch_i + 1)
        else:
            print(epoch_print)

        # save ----
        if do_log:
            save_dict = {'epoch': epoch_i + 1,
                         'arch': model.name,
                         'state_dict': model.state_dict(),
                         'best_acc1': best_val_score,
                         'optimizer': train_manager['optimizer'].state_dict()}
            logx.save_model(save_dict, metric=val_loss,
                            epoch=epoch_i + 1, higher_better=False)


def run_epoch(model, train_iter, train_manager, epoch_i=None, do_log=False):
    model.train()
    optimizer = train_manager['optimizer']
    loss_fn = train_manager['loss_fn']

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        inputs = batch['inp'].double().to(device)
        labels = batch['trg'].double().to(device)

        if model.name == 'informer':
            time_embd = batch['time_embd'].double().to(device)
            prediction = model(inputs, time_embd)
        else:
            prediction = model(inputs)

        # e.g. transformer model uses the dim: T x B x C
        if not model.batch_first:
            labels = labels.permute(1, 0, 2)

        loss = loss_fn(prediction, labels)
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), .7)  # also seen .5 elsewhere
        optimizer.step()

        # log results
        if do_log:
            if epoch_i is not None and i % 5 == 0:
                writer_path = f"Loss/train/{train_manager['loss_label']}/{train_manager['year_test']}"
                logx.add_scalar(writer_path, loss, epoch_i *
                                len(train_iter) + i)
            if i % 100 == 0:
                logx.msg(
                    f"\t[max val: {torch.max(prediction)}\tmin val, {torch.min(prediction)}]")

    return loss


def evaluate_validation(model, val_iter, train_manager):
    return evaluate_model(model, val_iter, train_manager)

# --- --- ---
# --- --- ---


if __name__ == "__main__":
    main()
