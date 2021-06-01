# --- --- ---
# pytorch_utils.py
# Sven Giegerich / 13.05.2021
# --- --- ---

import numpy as np
import pickle


# See: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.do_save_model = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.do_save_model = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.do_save_model = False
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.do_save_model = True
            self.counter = 0

    def save_checkpoint(self, val_loss, dict):
        '''Saves model when validation loss decrease.'''
        # SVEN
        # save dict via pickle instead of a model
        # only if path is given else the logx is used
        if self.path is not None:
            if self.verbose:
                self.trace_func(
                    f'>> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            pickle.dump(dict, open(self.path, 'wb'))
            self.val_loss_min = val_loss
