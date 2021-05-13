# --- --- ---
# train.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import torch

from libs.losses import *

def train(model, train_data, val_data):
    print("> Train")
    
    if model.name == 'tsmom':
        train_tsmom()
    
    pass

def train_epoch():
    pass

def train_tsmom():
    pass


