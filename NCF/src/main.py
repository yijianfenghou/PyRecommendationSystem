import os
import numpy as np
import argparse
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

import NCF.src.config as config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epoches", type=int, default=30, help="training epoches")
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--layers", nargs="+", default=[64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layer[0]/2 is the embedding size")
    parser.add_argument("--num_ng_test", type=int, default=100, help="Number of negative sample for test set")
    parser.add_argument("--out", default=True, help="save model or not")

    # set device and parameters
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    # seed for Reproducibility
    # util.seed_evenything(args.seed)

    # load data
    ml_1m = pd.read_csv(config.DATA_PATH, sep="::", names=['user_id', 'item_id', 'rating', 'timestamp'],
                        engine='python')

    # set the num_users, items
    num_users = ml_1m["user_id"].nunique() + 1
    num_items = ml_1m["item_id"].nunique() + 1

    # construct the train and test datasets
    
