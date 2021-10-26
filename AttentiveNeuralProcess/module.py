import torch as t
import torch.nn as nn
import math

class Linear(nn.Module):
    '''
    Linear Module
    '''
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        '''
        :param in_dim: dimension of input
        :param out_dim: demension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        '''
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_normal_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class MultiheadAttention(nn.Module):
    '''
    Multihead attention mechanism(dot attention)
    '''
    def __init__(self, num_hidden_k):
        '''
        :param num_hidden_k: dimension of hidden
        '''
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        attn = t.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn



class Attention(nn.Module):
    '''
    Attention Network
    '''
    def __init__(self, num_hidden, h=4):
        '''
        :param num_hidden: dimension of hidden
        :param h: num of heads
        '''
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden//h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden*2, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)



class LatentEncoder(nn.Module):
    '''
    Latent Encode [for prior, posterior]
    '''
    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = t.cat([x, y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            pass
