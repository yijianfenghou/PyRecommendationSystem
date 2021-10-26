import torch.nn as nn
import torch
import torch.utils.data
import numpy as np
import pandas as pd

class NCFData(torch.utils.data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        # Note that the labels are only useful when training, we thus add them in the ng_sample() function.
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        """
        if self.is_training:
            self.ng_sample()
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels
        """
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(GMF, self).__init__()
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weights_()

    def _init_weights_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF*embed_item_GMF
        prediction = self.predict_layer(output_GMF)

        return prediction.view(-1)

class MLP(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(MLP, self).__init__()
        self.embed_user_MLP = nn.Embedding(user_num, factor_num*(2**(num_layers-1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num*(2**(num_layers-1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2**(num_layers-i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())

        self.MLP_layers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(factor_num, 1)

        self._init_weights_()

    def __init_weights_(self):
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat(embed_user_MLP, embed_item_MLP)

        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)

        return prediction


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num*(2**(num_layers-1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num*(2**(num_layers-1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2**(num_layers-i))
            MLP_modules.append(nn.Dropout(dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(factor_num*2, 1)
        self._init_weights_()

    def _init_weights_(self):
        nn.init.normal_(self.embed_user_GMF, std=0.01)
        nn.init.normal_(self.embed_item_GMF, std=0.01)
        nn.init.normal_(self.embed_user_MLP, std=0.01)
        nn.init.normal_(self.embed_item_MLP, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):

        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

# loading dataset function
def load_dataset(test_num=100):

    train_data = pd.read_csv("../DataSets/ncf/")



