import torch
import torch.nn as nn
import numpy as np


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim, 1)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        x = x + x.new_temsor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FieldAwareFactorizationMachine, self).__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(field_dims[i], embed_dim) for i in range(self.num_fields)
        ])

        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embeddings in self.embeddings:
            nn.init.xavier_normal_(embeddings.weight.data)

    def forward(self, x):

        x = x + x.new_temsor(self.offsets, dtype=np.long).unsqueeze(0)
        xs = [self.embeddings[i](x[:, i]) for i in range(self.num_fields)]
        # xs = torch.cat(xs, dim=1)
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i+1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix



