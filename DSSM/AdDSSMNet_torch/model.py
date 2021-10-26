import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DSSMModel(nn.Module):

    def __init__(self, input1, input2, pic_embedding_size, batch_size, hidden_size, output_size):
        super(DSSMModel, self).__init__()

        # 用户和匹配共同的特征
        self.input1 = input1
        self.input2 = input2

        self.hidden_size = hidden_size
        self.output_size = output_size

        embedding_size = []

        for i, feat_size in enumerate(self.input1):
            emb, emb_size = self.create_feat_embedding(feat_size, max_embed_size=50)
            setattr(self, "embed_{}".format(i), emb)
            embedding_size.append(emb_size)

        self.linear1_1 = nn.Sequential(
            nn.Linear(sum(embedding_size), 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=0.2)
        )
        self.linear1_1.apply(self._init_weights)

        self.linear1_2 = nn.Sequential(
            nn.Linear(sum(embedding_size) + pic_embedding_size, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=0.2)
        )
        self.linear1_2.apply(self._init_weights)

        self.linear2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            #nn.Dropout(p=.2)
        )
        self.linear2.apply(self._init_weights)

        self.linear3 = nn.Sequential(
            nn.Linear(512, output_size),
            nn.ReLU(inplace=True)
        )
        self.linear3.apply(self._init_weights)

        self.last_linear = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def create_feat_embedding(self, feat_size, max_embed_size):
        embed_size = min((feat_size + 4) // 2, max_embed_size)
        emb = nn.Embedding(feat_size, embedding_dim=embed_size)
        emb.apply(self._init_weights)
        return emb, embed_size

    def forward_user_side(self, x):
        embed = [getattr(self, "embed_{}".format(i))(x[:, i]) for i in range(len(self.input1))]
        embed = torch.cat(embed, dim=1)
        out = self.linear1_1(embed)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def forward_match_user_side(self, x, pic_embedding):
        embed = [getattr(self, "embed_{}".format(i))(x[:, i]) for i in range(len(self.input1))]
        embed = torch.cat(embed, dim=1)
        embed = torch.cat((embed, pic_embedding), dim=1)
        embed = Variable(embed, requires_grad=False)
        # print(embed)

        out = self.linear1_2(embed)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def forward(self, input1, input2, pic_embedding):
        # pic_embedding = Variable(pic_embedding, requires_grad=False)
        pic_embedding = Variable(pic_embedding)
        out1 = self.forward_user_side(input1)
        out2 = self.forward_match_user_side(input2, pic_embedding)

        # 用户和被推荐人之间
        # dot_user_match_user = torch.sum(out1*out2, dim=1) / (torch.sqrt(torch.sum(out1**2))*torch.sqrt(torch.sum(out2**2)))
        dot_user_match_user = F.cosine_similarity(out1, out2, dim=1)
        # print(dot_user_match_user)
        print("--------------------------------------------")
        # dot_user_match_user = dot_user_match_user.unsqueeze(1)
        # print(dot_user_match_user)
        # output = dot_user_match_user
        # output = self.last_linear(dot_user_match_user)

        return dot_user_match_user
