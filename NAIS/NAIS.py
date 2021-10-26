import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="Run NAIS.")
parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='pinterest-20', help='Choose a dataset.')
parser.add_argument('--pretrain', type=int, default=1, help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
parser.add_argument('--batch_choice', nargs='?', default='user', help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
parser.add_argument('--weight_size', type=int, default=16, help='weight size.')
parser.add_argument('--embed_size', type=int, default=16, help='Embedding size.')
parser.add_argument('--data_alpha', type=float, default=0, help='Index of coefficient of embedding vector')
parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,1e-5]', help='Regularization for user and item embeddings.')
parser.add_argument('--alpha', type=float, default=0, help='Index of coefficient of embedding vector')
parser.add_argument('--train_loss', type=float, default=1, help='Caculate training loss or nor')
parser.add_argument('--beta', type=float, default=0.5, help='Index of coefficient of sum of exp(A)')
parser.add_argument('--num_neg', type=int, default=4, help='Number of negative instances to pair with a positive instance.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--activation', type=int, default=0, help='Activation for ReLU, sigmoid, tanh.')
parser.add_argument('--algorithm', type=int, default=0, help='0 for NAIS_prod, 1 for NAIS_concat')


class NAISNet(nn.Module):

    def __init__(self, args, num_items):
        super(NAISNet, self).__init__()
        self.pretrain = args.pretrain
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.weight_size = args.weight_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_alpha = args.data_alpha
        self.verbose = args.verbose
        self.activation = args.activation
        self.algorithm = args.algorithm
        self.batch_choice = args.batch_choice
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.train_loss = args.train_loss


    def forward(self, x):
        pass


if __name__ == "__mian__":
    pass

