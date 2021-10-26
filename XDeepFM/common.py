import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class FirstOrder(nn.Module):
    def __init__(self, params):
        super(FirstOrder, self).__init__()
        self.device = params['device']
        self.feature_size = params['feature_size']
        weights_first_order = torch.empty(self.feature, 1, dtype=torch.float32, device=self.device, requires_grad=True)
        nn.init.normal_(weights_first_order)
        self.weights_first_order = nn.Parameter(weights_first_order)

    def forward(self, feature_values, feature_idx):
        weights_first_order = self.weights_first_order[feature_idx, :]
        first_order = torch.mul(feature_values, weights_first_order.squeeze())
        return first_order

class SecondOrder(nn.Module):
    def __init__(self, params, get_embeddings=False):
        super(SecondOrder, self).__init__()
        self.device = params['device']
        self.feature_size = params['feature_size']
        self.embedding_size = params['embedding_size']
        self.get_embeddings = get_embeddings
        feature_embeddings = torch.empty(self.feature_size, self.embedding_size, dtype=torch.float32, device=self.device, requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)

    def forward(self, feature_values, feature_idx):
        embeddings = self.feature_embeddings[feature_idx,:]
        temp1 = torch.pow(torch.einsum('bf,bfk->bk', (feature_values, embeddings)), 2)
        temp2 = torch.einsum('bf,bfk->bk', (torch.pow(feature_values, 2), torch.pow(embeddings, 2)))
        second_order = temp1 - temp2
        if self.get_embeddings:
            return second_order, embeddings
        else:
            return second_order


class MLP(nn.Module):
    def __init__(self, params, use_batchnorm=True, use_dropout=True):
        
        self.embedding_size = params['embedding_size']
        self.field_size = params['field_size']
        self.hidden_dims = params['hidden_dims']
        self.device = params['device']
        self.p = params['p']
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.input_dim = self.field_size*self.embedding_size
        self.num_layers = len(self.hidden_dims)
        
        ## deep weights
        self.deep_layers = nn.Sequential()
        
        net_dims = [self.input_dim]+self.hidden_dims
        for i in range(self.num_layers):
            self.deep_layers.add_module('fc%d'%(i+1), nn.Linear(net_dims[i], net_dims[i+1]).to(self.device))
            if self.use_batchnorm:
                self.deep_layers.add_module('bn%d'%(i+1), nn.BatchNorm1d(net_dims[i+1]).to(self.device))
            self.deep_layers.add_module('relu%d'%(i+1), nn.ReLU().to(self.device))
            if self.use_dropout:
                self.deep_layers.add_module('dropout%d'%(i+1), nn.Dropout(self.p).to(self.device))

    def forward(self, embeddings):
        deepInput = embeddings.reshape(embeddings.shape[0], self.input_dim)
        deepOut = self.deep_layers(deepInput)
        return deepOut

class CIN(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super(CIN, self).__init__()

        # CIN网络有几层，也就是要几阶
        self.num_layers = num_layers
        # 一维卷积层
        self.conv_layers = torch.nn.ModuleList()
        fc_input_dim = 0
        for i in range(self.num_layers):
            """
            in_channels: 输入信号的通道 向量的维度，input_dim的长度指的是特征的总数
            output_channels: 卷积产生的通道。有多少个out_channels,就需要多少个1维卷积
            kernal_size： 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernal_size*in_channels
            stride: 卷积步长
            dilation： 卷积核元素之间的间距
            """
            temp1 = torch.nn.Conv1d(in_channels=input_dim*input_dim, out_channels=input_dim, kernel_size=1, stride=1, dilation=1, bias=True)
            self.conv_layers.append(temp1)
            fc_input_dim += input_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        """
        举例 x.shape = [1, 22, 16] 1表示batch_size，表示有几维数据，22表示特征的维数，16是embedding层的向量大小
        经过x.unsqueeze(2)后x.shape = [1, 22, 1, 16]
        经过x.unsqueeze(1)后x.shape = [1, 1, 22, 16]
        x.unsqueeze(2)*x.unsqueeze(1)后 x.shape = [1, 22, 22, 16]
        进过卷积层后变成为x.shape = [1, 16, 16]
        经过sum pooling变成1维
        """
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            h1 = h.unsqueeze(1)
            x = x0*h1
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim*fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))