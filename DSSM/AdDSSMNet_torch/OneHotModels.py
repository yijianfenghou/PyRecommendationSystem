import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DSSMModel(nn.Module):

    def __init__(self, user_fields_dim, item_fields_dim, max_embed_size, hidden_size, output_size):
        super(DSSMModel, self).__init__()

        self.out1 = None
        self.out2 = None

        self.num_user_fields = len(user_fields_dim)
        self.num_item_fields = len(item_fields_dim)

        self.embed_user_data = [nn.Embedding(user_fields_dim[i], min((user_fields_dim[i] + 4) // 2, max_embed_size)) for
                                i in range(self.num_user_fields)]
        self.embed_item_data = [nn.Embedding(item_fields_dim[i], min((item_fields_dim[i] + 4) // 2, max_embed_size)) for
                                i in range(self.num_item_fields)]

        embed_user_size = [min((user_fields_dim[i] + 4) // 2, max_embed_size) for i in range(self.num_user_fields)]
        embed_item_size = [min((item_fields_dim[i] + 4) // 2, max_embed_size) for i in range(self.num_item_fields)]

        self.user_linear = nn.Sequential(
            nn.Linear(sum(embed_user_size), hidden_size[0]),
            nn.ReLU(inplace=True)
        )

        self.item_linear = nn.Sequential(
            nn.Linear(sum(embed_item_size), hidden_size[0]),
            nn.ReLU(inplace=True)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(inplace=True)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(hidden_size[1], output_size)
        )

        # self._init_weight_()

    # def _init_weight_(self):
    #     nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
    #     nn.init.normal_(self.embed_match_user_GMF.weight, std=0.01)

    # def _get_input_index_(self, x):
    #     x_ = torch.nonzero(x, as_tuple=False).view(self.batch_size, self.filed_dims, -1)[:, :, 1]
    #     return x_

    def forward_user_side(self, x):

        out = [self.embed_user_data[i](x[:, i]) for i in range(self.num_user_fields)]
        out = torch.cat(out, dim=1)

        out = self.user_linear(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def forward_match_user_side(self, x):
        out = [self.embed_item_data[i](x[:, i]) for i in range(self.num_item_fields)]
        out = torch.cat(out, dim=1)

        out = self.item_linear(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def cosine(self, input1, input2):
        input1_norm = torch.norm(input1, dim=1, keepdim=True)
        input2_norm = torch.norm(input2, dim=1, keepdim=True)

        cosine = torch.sum(torch.mul(input1_norm, input2_norm), dim=1, keepdim=True) / (input1_norm * input2_norm)
        return cosine

    def forward(self, user_input, item_input):
        user_input = Variable(user_input)
        out1 = self.forward_user_side(user_input)
        out2 = self.forward_match_user_side(item_input)
        self.out1 = out1
        self.out2 = out2
        # user_match_user_sim = (1 - F.cosine_similarity(out1, out2, dim=1)) / 2
        # user_match_user_sim = F.pairwise_distance(out1, out2)
        # pos_result = self.cosine(out1, out2)
        # neg_result = 1 - pos_result
        # logits = torch.cat([pos_result, neg_result], dim=1)
        return out1, out2
