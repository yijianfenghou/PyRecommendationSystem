import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

class MyDataSet(nn.Module):

    def __init__(self, labelPath, idxPath, valuePath, batch_size):
        label = np.loadtxt(labelPath, delimiter='\t')
        idx = np.loadtxt(idxPath, delimiter='\t')
        value = np.loadtxt(valuePath, delimiter='\t')

        data = np.concatenate((label, idx, value), 1)

        self.len = self.data.shape[0]

        self.data = torch.from_numpy(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_test_model_demo(train_label_path, train_idx_path, train_value_path):
    # 读取数据方式  ，数据为大文件时，节省空间，效率训练
    def get_batch_dataset(label_path, idx_path, value_path):
        myDataSet = MyDataSet(train_label_path, train_idx_path, train_value_path)

        setup_seed(20)

        train_dataset = TensorDataset(label, idx, value)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=32, num_workers=2)

        return train_loader

    train_batch_datasets = get_batch_dataset(train_label_path, train_idx_path, train_value_path)





if __name__ == "__main__":
    dataset = MyDataSet("XXX")
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)