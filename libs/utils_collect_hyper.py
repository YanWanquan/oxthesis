# --- --- ---
# collect_hyperparams.py
# Sven Giegerich / 02.06.2021
# --- --- ---

import os
import argparse
import glob
import pandas as pd
from functools import reduce


def get_args():
    parser = argparse.ArgumentParser(
        description='Helps to collect the results of hyperparams search')

    parser.add_argument('--dir', type=str, nargs='?',
                        help="Directory of a experiment")
    parser.add_argument('--save_dir', type=str, nargs='?', default="results/",
                        help="Directory of a experiment")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    files = glob.glob(args.dir + "**/exp_win_test_loss_*.csv", recursive=True)
    dfs = []
    for i, path in enumerate(files):
        dfs.append(pd.read_csv(path, index_col=0))
    df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True,
                                             how='outer'), dfs)
    df.index.name = "time"

    name_experiment = args.dir.split("/")[-2]
    save_path = os.path.join(args.save_dir, f"log_{name_experiment}.csv")
    print(f"> Save file to {save_path}")
    df.to_csv(save_path)


if __name__ == "__main__":
    main()
