# --- --- ---
# train.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import libs.utils as utils
from datetime import datetime
import torch
from libs.losses import LossHelper, LossTypes
from libs.losses import *
from torch.utils.tensorboard import SummaryWriter

import evaluate
from libs.pytorch_utils import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_data, val_data, train_details):
    print("> Train")
    
    # skip training for the simple strategies
    if model.name in ['tsmom', 'long']:
        train_tsmom()
        return 1

    writer_path = f".runs/{model.name}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    writer = SummaryWriter(writer_path)

    model = model.double()
    
    # training details ----
    train_details['optimizer'] = torch.optim.AdamW(model.parameters(), lr=train_details['lr'])
    check_point_path = utils.get_save_path(file_label="checkpoint", model=model.name, time_test=train_details['time_test'], file_type="pt", loss_type=train_details['loss_type'])
    early_stopping = EarlyStopping(patience=train_details['patience'], path=check_point_path, verbose=False)

    # start training ----
    for epoch_i in range(train_details['epochs']):
        epoch_loss = run_epoch(model, train_data, train_details, writer=writer, epoch_i=epoch_i)
        val_loss = evaluate_validation(model, val_data, train_details)

        # verb ----
        print(f">> Epoch Step: {epoch_i}\t val loss: {val_loss}\t train loss: {epoch_loss}")
        writer.add_scalar("Loss/val", val_loss, epoch_i)

        # early stopping ----
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"> Early stopping")
            break
    
    # load last checkpoint with best model
    model.load_state_dict(torch.load(check_point_path))

    writer.close()

    return check_point_path


def run_epoch(model, train_iter, train_details, verb=True, writer=None, epoch_i=None):
    model.train()
    optimizer = train_details['optimizer']
    loss_fn = train_details['loss_fn']

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        inputs = batch['inp'].double().to(device)
        labels = batch['trg'].double().to(device)

        prediction = model(inputs)

        # transformer model uses the dim: T x B x C
        if model.name == 'transformer':
            labels = labels.permute(1, 0, 2)

        loss = loss_fn(prediction, labels)
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), .7) # also seen .5 elsewhere
        optimizer.step()

        # log results
        if writer is not None and epoch_i is not None and i % 5 == 0:
            print(torch.max(prediction), torch.min(prediction))
            writer.add_scalar("Loss/train", loss, epoch_i * len(train_iter) + i)

    return loss

def evaluate_validation(model, val_iter, train_details):
    return evaluate.evaluate_model(model, val_iter, train_details)

def train_tsmom():
    pass


