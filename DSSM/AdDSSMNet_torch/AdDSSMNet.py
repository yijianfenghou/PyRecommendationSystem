import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

batch_size = 64


class AdDSSMNet(nn.Module):

    def __init__(self, cat_sizes=None, embedding_factor=3, out_dims=32):
        super(AdDSSMNet, self).__init__()
        self.cat_sizes = cat_sizes
        self.embedding_factor = embedding_factor

        embedding_sizes = []

        for i, cat_size in enumerate(self.cat_sizes):
            emb, emb_size = self.create_emb(cat_size=cat_size, max_emb_size=50)
            setattr(self, "emb_{}".format(i), emb)
            embedding_sizes.append(emb_size)

        total_embedding_sizes = sum(embedding_sizes)
        self.linear1 = nn.Sequential(
            nn.Linear(total_embedding_sizes, 128),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64, out_dims),
            nn.ReLU(inplace=True)
        )

    def create_emb(self, cat_size, max_emb_size):
        emb_size = min([(cat_size + 4) // self.embedding_factor, max_emb_size])
        emb = nn.Embedding(num_embeddings=cat_size, embedding_dim=emb_size)
        return emb, emb_size

    def forward_once(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, input1, input2):
        embed1 = [getattr(self, "emb_{}".format(i))(input1[:, i]) for i, cat_size in enumerate(self.cat_sizes)]
        embed1 = torch.cat(embed1, dim=1)
        embed2 = [getattr(self, "emb_{}".format(i))(input2[:, i]) for i, cat_size in enumerate(self.cat_sizes)]
        embed2 = torch.cat(embed2, dim=1)

        out1 = self.forward_once(embed1)
        out2 = self.forward_once(embed2)
        return out1, out2


if __name__ == "__main__":
    df = pd.read_excel("C:/Users/EDZ/Desktop/result.xlsx")
    # 处理用户数据
    df.dropna(subset=['subeventid'], inplace=True)
    # df['subeventid'] = df['subeventid'].fillna('0')
    # df['subeventid'] = df['subeventid'].dropna(axis=0)
    df['subeventid'] = df['subeventid'].astype('int64')
    uidSet = set(df["uid"].tolist() + df["subeventid"].tolist())
    uidToIdx = {key: i for i, key in enumerate(uidSet)}

    df['age'] = df['age'].fillna(0)
    df['opposite_age'] = df['opposite_age'].fillna('0')
    df['opposite_age'] = df['opposite_age'].astype('int64')
    uidAgeSet = set(df["age"].tolist() + df["opposite_age"].tolist())
    uidAgeIdx = {key: i for i, key in enumerate(uidAgeSet)}

    df['height'] = df['height'].fillna(0)
    df['opposite_height'] = df['opposite_height'].fillna('0')
    df['opposite_height'] = df['opposite_height'].astype('int64')
    uidHeightSet = set(df["height"].tolist() + df["opposite_height"].tolist())
    uidHightIdx = {key: i for i, key in enumerate(uidHeightSet)}

    df['sex'] = df['sex'].fillna(0)
    df['opposite_sex'] = df['opposite_sex'].fillna('0')
    uidSexSet = set(df["sex"].tolist() + df["opposite_sex"].tolist())
    uidSexIdx = {key: i for i, key in enumerate(uidSexSet)}

    labels = torch.FloatTensor(df['label'])

    cat_sizes = [len(uidToIdx), len(uidAgeIdx), len(uidHightIdx), len(uidSexIdx)]

    model = AdDSSMNet(cat_sizes)

    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    df["uid"] = df["uid"].apply(lambda x: uidToIdx[x])
    df["subeventid"] = df["subeventid"].apply(lambda x: uidToIdx[x])
    df["age"] = df["age"].apply(lambda x: uidAgeIdx[x])
    df["opposite_age"] = df["opposite_age"].apply(lambda x: uidAgeIdx[x])
    df["height"] = df["height"].apply(lambda x: uidHightIdx[x])
    df["opposite_height"] = df["opposite_height"].apply(lambda x: uidHightIdx[x])
    df["sex"] = df["sex"].apply(lambda x: uidSexIdx[x])
    df["opposite_sex"] = df["opposite_sex"].apply(lambda x: uidSexIdx[x])

    uidData = torch.from_numpy(df[["uid", "age", "height", "sex"]].values)
    tuidData = torch.from_numpy(df[["subeventid", "opposite_age", "opposite_sex", "opposite_sex"]].values)

    train_dataset = TensorDataset(uidData, tuidData, labels)
    train_dataset = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(2):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_dataset):
            uid_data, tuid_data, label_data = data
            uid_input = Variable(uid_data)
            tuid_input = Variable(tuid_data)
            label = Variable(label_data)

            # 梯度置零
            optimizer.zero_grad()
            out1, out2 = model(uid_input, tuid_input)

            cos_dis = torch.sigmoid(F.cosine_similarity(out1, out2, dim=1))

            loss = criterion(cos_dis, label)

            # 反向传播
            loss.backward(torch.ones_like(loss))
            optimizer.step()

            running_loss += loss.sum()
            if i % 2 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0