# --- --- ---
# train.py
# Sven Giegerich / 03.05.2021
# --- --- ---

from datetime import datetime
import torch
from libs.losses import LossHelper, LossTypes
from libs.losses import *
from torch.utils.tensorboard import SummaryWriter

import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_data, val_data, train_details):
    print("> Train")
    
    # skip training for the simple strategies
    if model.name in ['tsmom', 'long']:
        train_tsmom()
        pass

    writer = SummaryWriter(f".runs/{model.name}/{datetime.now().strftime('%Y_%m_%d-%H_%M')}")

    model = model.double()
    
    # add training 
    train_details['optimizer'] = torch.optim.AdamW(model.parameters(), lr=train_details['lr'])

    for epoch_i in range(1, train_details['epochs']+1):
        epoch_loss = run_epoch(model, train_data, train_details)
        val_loss = evaluate_validation(model, val_data, train_details)

        print(f">> Epoch Step: {epoch_i}\t val loss: {val_loss}\t train loss: {epoch_loss}")
        writer.add_scalar("Loss/train", epoch_loss, epoch_i)
        writer.add_scalar("Loss/val", val_loss, epoch_i)

    writer.close()


def run_epoch(model, train_iter, train_details, verb=True):
    model.train()
    optimizer = train_details['optimizer']
    loss_fn = train_details['loss_fn']()

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

        break

    return loss

def evaluate_validation(model, val_iter, train_details):
    return evaluate.evaluate_model(model, val_iter, train_details)




            


            


def train_tsmom():
    pass


