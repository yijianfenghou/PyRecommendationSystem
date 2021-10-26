import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset


# example:  设样本有三个field，num_f  ield = 3，field1取值有10种情况，field2取值有20种情况，field3取值有10种情况
# 那么field_dims=[10,20,10],令嵌入维度embed_dim=4
# 在forward中，由于一次读取的是batch_size个数据，设batch_size=5
# 那么x = [[1,3,1], [1,7,1], [2,10,2], [3,10,3], [4,11,2]]   x的shape为：batch_size*num_field=5*3

class FieldAwareFactorizationMachineModel(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FieldAwareFactorizationMachineModel, self).__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        # 其中线性部分是w0+sum(wi*xi), ffm部分是sum(sum((vi,f*vj,f)*xi*xj))
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))


# sum(field_dims) = 10+20+10 = 40, output_dim嵌入维度默认为1
# self.fc = torch.nn.Embedding(sum(field_dims), output_dim)相当于构建了一个索引字典,
# 索引为1到40,每个索引对应一个长度为output_dim=1的向量
# bias就是公式中的w0
# 为什么需要self.offsets,是这样的：
# 以样本[1,3,1]为例,one-hot编码过后其实是:
# [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
# 那么"1"所在的位置对应的索引分别为1、13、31.那该怎么得到这个索引呢，offsets发挥作用了,
# 因为eg filed_dims=[10, 20,10],那麽offsets=[0,10,30]，把[1, 3, 1] + [0, 10, 20] = [1, 13, 31]
# 因为输入的x = [[1,3,1], [1,7,1], [2,10,2], [3,10,3], [4,11,2]]
# 所以通过加上offsets之后 x变为了[1,13,31],[1,17,31],[2,20,32],[3,20,33],[4,21,32]]
# self.fc(x)的会得到一个batch_size*num_field*output_dim的tensor（不清楚可以查看pytorch中embdding的用法),
# tensor([[[-0.3187], [-0.1316], [ 0.1061]],
#         [[-0.3187], [ 0.1420], [ 0.1061]],
#         [[-0.2323], [ 0.1549], [ 0.2619]],
#         [[ 0.2500], [ 0.1549], [ 0.0837]],
#         [[ 0.1705],  [ 0.2401],[ 0.2619]]]
# [1,13,31]对应[[-0.3187], [-0.1316], [ 0.1061]],大小为num_filed*output_dim=3*1
# [1,17,31]对应[[-0.3187], [ 0.1420], [ 0.1061]],大小为num_filed*output_dim=3*1
# torch.sum(self.fc(x), dim=1)得到一个batch_size*output_dim大小的张量：
# 对于[1,13,31]就是把[[-0.3187], [-0.1316], [ 0.1061]]第dim=1维的数据相加.变成为[-0.3187]+[-0.1316]+ [0.1061]=[-0.3534]

class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        print("FeaturesLinear x=", x)
        x = x + x.new_tensor(self.offsets, dtype=np.long).unsqueeze(0)
        print("FeaturesLinear return=", torch.sum(self.fc(x), dim=1) + self.bias)
        return torch.sum(self.fc(x), dim=1) + self.bias


# 因为在FFM中,每一维特征 xi,针对其它特征的每一种field fj,都会学习一个隐向量 v_i,fj，
# 所以有多少个field们就要构建多少个torch.nn.Embedding层
# （不太会用术语解释，相当于总的特征为sum(field_dims)=40, 嵌入维度embed_dim=4,
# 共有num_field=3个field，所以就要构建3个embedding,
# 当i=1，f_{j}=2，得到v_{1,2},表示特征1对第2个field的一个长度为4的向量）
# offset的作用和上文提到的一样
# torch.nn.init.xavier_uniform_(embedding.weight.data)是一种初始化嵌入层权重的方法
# xs = [self.embeddings[i](x) for i in range(self.num_fields)]
# 将会的到长度为num_field=3的列表，列表中的每一个元素，大小都为batch_size*num_field*embed_dim=5*3*4
# 形如：（[1,13,31],[1,17,31],[2,20,32],[3,20,33],[4,21,32]]）
# 列表的第1个元素分别记录着当前batch中的所有样本各自的特征对第1个field1的隐向量
# 列表的第2个元素分别记录着当前batch中的所有样本各自的特征对第2个field2的隐向量
# 列表的第3个元素分别记录着当前batch中的所有样本各自的特征对第3个field3的隐向量

class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FieldAwareFactorizationMachine, self).__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding.weight.data)

    def forward(self, x):

        x = x + x.new_tensor(self.offsets, dtype=np.long).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    print(dataset[:10])
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = FieldAwareFactorizationMachineModel(dataset.field_dims, embed_dim=4).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='XXX')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
