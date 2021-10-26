import torch
import torch.nn as nn

import pickle


class BiInteraction(nn.Module):

    def __init__(self, Units = 1):
        super(BiInteraction, self).__init__()
        self.units = Units

    def forward(self, input):
        # sum-square-part
        self.summed_features_emb = torch.sum(input, dim=1)   # batch_size * K
        # print("self.summed_features_emb:",self.summed_features_emb.size())
        self.summed_features_emb_square = torch.square(self.summed_features_emb)

        # square-sum-part
        self.squared_features_emb = torch.square(input)
        self.squared_sum_features_emb = torch.sum(self.squared_features_emb)

        # second order
        self.y_second_order = 0.5*torch.sub(self.summed_features_emb_square, self.squared_sum_features_emb)  # batch_size * K
        print("y_second_order:", self.y_second_order.size())
        output = nn.Linear(self.y_second_order.size()[1], self.units)(self.y_second_order)
        return output


class NFM(nn.Module):

    def __init__(self, num_feats, num_fields, num_hidden_layers, deep_layer_sizes, dropout_deep, embedding_size):
        super(NFM, self).__init__()
        self.num_feats = num_feats    # F = features nums
        self.num_fields = num_fields  # N = fields of a feature
        self.dropout_deep = dropout_deep

        # Embedding 这里采用embeddings层因此大小为F* M F为特征数量，M为embedding的维度
        feat_embeddings = nn.Embedding(num_feats, embedding_size)
        self.feat_embeddings = feat_embeddings

        # linear layer
        self.deep_layer_sizes = deep_layer_sizes

        # 神经网络方面的参数
        for i in range(len(deep_layer_sizes)):
            setattr(self, "linear_{}".format(i), nn.Linear(deep_layer_sizes[i][0], deep_layer_sizes[i][1]))
            setattr(self, "batchNorm_{}".format(i), nn.BatchNorm1d(deep_layer_sizes[i][1]))
            setattr(self, "activation_{}".format(i), nn.ReLU())
            setattr(self, "dropout_{}".format(i), nn.Dropout(dropout_deep[i]))

        # 相互交叉层
        self.bilayer = BiInteraction(1)

        # last linear
        self.last_linear = nn.Linear(deep_layer_sizes[-1][1], 1, bias=True)
        self.linearlayer = nn.Sequential(
            nn.Linear(deep_layer_sizes[-1][0], deep_layer_sizes[-1][1], bias=True),
            nn.ReLU()
        )


    def forward(self, feat_index, feat_value):

        # embedding part feat_index = inputs 为输入feat_embeddings 为一个layers
        feat_embedding_0 = self.feat_embeddings(feat_index)
        # print(feat_value.size())
        feat_embedding = torch.einsum("bnm,bn->bnm", feat_embedding_0, feat_value)

        y_deep = self.bilayer(feat_embedding)
        y_linear = self.linearlayer(torch.sum(feat_embedding, 1))

        for i in range(len(self.deep_layer_sizes)):
            y_deep = getattr(self, 'linear_{}'.format(i))(y_deep)
            y_deep = getattr(self, "batchNorm_{}".format(i))(y_deep)
            y_deep = getattr(self, "activation_{}".format(i))(y_deep)
            y_deep = getattr(self, "dropout_{}".format(i))(y_deep)

        y = y_deep + y_linear

        output = self.last_linear(y)
        return output

if __name__ == "__main__":

    AID_DATA_DIR = "../data/Criteo/"
    feat_dict = pickle.load(open(AID_DATA_DIR+'/feat_dict_10.pkl2', 'rb'))

    nfm = NFM(num_feat=len(feat_dict)+1, num_fields=39, dropout_deep=[0.5, 0.5, 0.5], deep_layer_sizes=[400, 400], embedding_size=10)

    train_label_path = AID_DATA_DIR + 'train_label'
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    train





