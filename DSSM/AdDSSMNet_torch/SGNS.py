import torch
import random
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

class SGNS(nn.Module):

    def __init__(self, vocab_size, projection_dim):

        super(SGNS, self).__init__()

        self.embedding_v = nn.Embedding(vocab_size, projection_dim)    # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)    # out embedding
        self.log_sigmoid = nn.LogSigmoid()

        init_range = (2.0 / (vocab_size + projection_dim))**0.5        # Xavier init
        self.embedding_v.weight.data.uniform_(-init_range, init_range) # init
        self.embedding_u.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words)                # B*1*D
        target_embeds = self.embedding_u(target_words)                # B*1*D
        neg_embeds = -self.embedding_u(negative_words)                 # B*K*D

        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)   # B*1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(center_words.size(0), -1) # B*K ---> B*1
        los = self.log_sigmoid(positive_score) + self.log_sigmoid(negative_score)

        return -torch.mean(los)

    # 获取单词的embedding
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        return embeds

class MyDataSets(nn.Module):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass

class SimpleNet(nn.Module):

    def __init__(self, in_dims, out_dims):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_dims, 128),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64, out_dims),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2

def get_batch_sample(bat_size, tra_data):
    random.shuffle(tra_data)     # 随机打乱数据
    s_index = 0
    e_index = bat_size

    while e_index < len(tra_data):
        bat = tra_data[s_index: e_index]
        temp = e_index
        e_index = e_index + bat_size
        s_index = temp
        yield bat

    if e_index >= len(tra_data):
        bat = tra_data[s_index:]
        yield bat

# 生成正样本
def get_positive_sample(samp_lists):
    positive_samples = []

    for sublist in samp_lists:
        sublist_length = len(sublist)
        for ite in sublist:
            ite_index = sublist.index(ite)
            for j in range(sublist_length):
                if ite_index != j:
                    positive_samples.append([ite, sublist[j]])
    target_words = []
    context_words = []

    for word_pair in positive_samples:
        target_words.append(word_pair[0])
        context_words.append(word_pair[1])

    return target_words, context_words   # 一维列表

# 生成负样本
def get_negative_sample(centers, targets, un_table, dict, k):
    batch_size = len(targets)   # 批次大小
    negative_samples = []

    for i in range(batch_size):
        neg_sample = []
        center_index = centers[i][0]
        target_index = centers[i][0]

        while len(neg_sample) < k:
            neg = random.choice(un_table)
            if neg == target_index or neg == center_index:
                continue
            neg_sample.append(dict[neg])

        negative_samples.append(neg_sample)

    # 返回一个二维列表
    return negative_samples

if __name__ == "__main__":

    movie_lists = []

    df = pd.read_excel("C:/Users/EDZ/Desktop/result.xlsx")
    # 对用户编号的处理
    df['subeventid'] = df['subeventid'].fillna('0')
    df['subeventid'] = df['subeventid'].astype('int64')
    uidSet = set(df["uid"].tolist() + df["subeventid"].tolist())
    uidToIdx = {key:i for i,key in enumerate(uidSet)}
    uidEmbedding = torch.nn.Embedding(len(uidToIdx), 32)
    uidEmbedding = uidEmbedding(Variable(torch.LongTensor(range(len(uidToIdx))))).data
    uidEmbeddingDict = {key: uidEmbedding[uidToIdx[key]] for key in uidToIdx}
    # df['uid'] = list(map(lambda x: uidEmbeddingDict[x], df['uid'].values))
    # df['subeventid'] = list(map(lambda x: uidEmbeddingDict[x], df['subeventid'].values))
    uid = list(map(lambda x: uidEmbeddingDict[x].cpu().numpy(), df['uid'].values))
    subeventid = list(map(lambda x: uidEmbeddingDict[x].cpu().numpy(), df['subeventid'].values))

    # 对用户年龄的处理
    df['age'] = df['age'].fillna(0)
    df['opposite_age'] = df['opposite_age'].fillna('0')
    df['opposite_age'] = df['opposite_age'].astype('int64')
    uidAgeSet = set(df["age"].tolist() + df["opposite_age"].tolist())
    uidAgeIdx = {key: i for i, key in enumerate(uidAgeSet)}
    uidAgeEmbedding = torch.nn.Embedding(len(uidAgeIdx), 2)
    uidAgeEmbedding = uidAgeEmbedding(Variable(torch.LongTensor(range(len(uidAgeIdx))))).data
    uidAgeEmbeddingDict = {key: uidAgeEmbedding[uidAgeIdx[key]] for key in uidAgeIdx}
    # df['age'] = list(map(lambda x: uidAgeEmbeddingDict[x], df['age'].values))
    # df['opposite_age'] = list(map(lambda x: uidAgeEmbeddingDict[x], df['opposite_age'].values))
    age = list(map(lambda x: uidAgeEmbeddingDict[x].cpu().numpy(), df['age'].values))
    opposite_age = list(map(lambda x: uidAgeEmbeddingDict[x].cpu().numpy(), df['opposite_age'].values))

    # 对用户身高
    df['height'] = df['height'].fillna(0)
    df['opposite_height'] = df['opposite_height'].fillna('0')
    df['opposite_height'] = df['opposite_height'].astype('int64')
    uidHeightSet = set(df["height"].tolist() + df["opposite_height"].tolist())
    uidHightIdx = {key: i for i, key in enumerate(uidHeightSet)}
    uidHightEmbedding = torch.nn.Embedding(len(uidHightIdx), 10)
    uidHightEmbedding = uidHightEmbedding(Variable(torch.LongTensor(range(len(uidHightIdx))))).data
    uidHightEmbeddingDict = {key: uidHightEmbedding[uidHightIdx[key]] for key in uidHightIdx}
    # df['height'] = list(map(lambda x: uidHightEmbeddingDict[x], df['height'].values))
    # df['opposite_height'] = list(map(lambda x: uidHightEmbeddingDict[x], df['opposite_height'].values))
    height = list(map(lambda x: uidHightEmbeddingDict[x].cpu().numpy(), df['height'].values))
    opposite_heigh = list(map(lambda x: uidHightEmbeddingDict[x].cpu().numpy(), df['opposite_height'].values))

    # 对性别处理
    df['sex'] = df['sex'].fillna(0)
    df['opposite_sex'] = df['opposite_sex'].fillna('0')
    #df['opposite_sex'] = df['opposite_sex'].astype('int64')
    uidSexSet = set(df["sex"].tolist() + df["opposite_sex"].tolist())
    uidSexIdx = {key: i for i, key in enumerate(uidSexSet)}
    uidSexEmbedding = torch.nn.Embedding(len(uidSexIdx), 2)
    uidSexEmbedding = uidSexEmbedding(Variable(torch.LongTensor(range(len(uidSexIdx))))).data
    uidSexEmbeddingDict = {key: uidSexEmbedding[uidSexIdx[key]] for key in uidSexIdx}
    # df['sex'] = list(map(lambda x: uidSexEmbeddingDict[x], df['sex'].values))
    # df['opposite_sex'] = list(map(lambda x: uidSexEmbeddingDict[x], df['opposite_sex'].values))
    sex = list(map(lambda x: uidSexEmbeddingDict[x].cpu().numpy(), df['sex'].values))
    opposite_sex = list(map(lambda x: uidSexEmbeddingDict[x].cpu().numpy(), df['opposite_sex'].values))

    labels = torch.FloatTensor(df['label'])

    uid = torch.Tensor(uid)
    subeventid = torch.Tensor(subeventid)
    age = torch.Tensor(age)
    opposite_age = torch.Tensor(opposite_age)
    height = torch.Tensor(height)
    opposite_heigh = torch.Tensor(opposite_heigh)
    sex = torch.Tensor(sex)
    opposite_sex = torch.Tensor(opposite_sex)

    uidEmbed = torch.cat((uid, age, height, sex), 1)
    tuidEmbed = torch.cat((subeventid, opposite_age, opposite_heigh, opposite_sex), 1)

    # 设置随机数种子
    # setup_seed(20)
    # uid_loader = DataLoader(uidEmbed, batch_size=64, shuffle=True)
    # tuid_loader = DataLoader(tuidEmbed, batch_size=64, shuffle=True)
    # simlabels = DataLoader(labels, batch_size=64, shuffle=True)

    train_dataset = TensorDataset(uidEmbed, tuidEmbed, labels)

    model = SimpleNet(46, 32)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model1 = model.cuda()

    # loss和optim
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # batch_size = 64
    # num_epochs = int(uidEmbed.size()[0]//batch_size)
    for epoch in range(1):

        running_loss = 0.0
        running_acc = 0.0
        for data in train_dataset:
            uid_loader, tuid_loader, label = data
            uid_input = Variable(uid_loader)
            tuid_input = Variable(tuid_loader)
            label = Variable(label)

            out1, out2 = model(uid_input, tuid_input)

            cos_dis = F.cosine_similarity(out1, out2, dim=1)
            loss = criterion(F.sigmoid(cos_dis), label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))  # 每2000次迭代，输出loss的平均值
                running_loss = 0.0










#     with open(r'result.xlsx', 'r', encoding='utf-8') as f:
#         contents = f.readlines()
#         for content in contents:
#             content = content.strip().split(",")
#             if content[0] == "":
#                 continue
#             movie_list = [int(m) for m in content]
#             if len(movie_list) > 1:
#                 movie_lists.append(movie_list)
#
#     fla = lambda k: [i for sublist in k for i in sublist]
#     item_counter = Counter(fla(movie_lists))
#     item = [w for w,c in item_counter.items()]
#
#     item2index = {}
#     for vo in item:
#         if item2index.get(vo) is None:
#             item2index[vo] = len(item2index)
#
#     item2index = {v: k for k,v in item2index.items()}
#
#     new_movie_lists = []
#     for m in movie_lists:
#         m = [item2index[n] for n in m]
#         new_movie_lists.append(m)
#
#     cent_words, cont_words = get_positive_sample(new_movie_lists)  # 一维列表
#
#     uni_table = []
#     f = sum([item_counter[it]**0.75 for it in item])     # 这边有个地方索引没转过来
#     z = 0.0001
#     for it in item:
#         uni_table.extend([it]*int(((item_counter[it]**0.75)/f)/z))
#
#     train_data = [[cent_words[i], cont_words[i]] for i in range(len(cont_words))]
#     item2vec = SGNS(len(item), 10)
#     optimizer = optim.adam(item2vec.parameters(), lr=0.001)
#
#     for epoch in range(1):
#         for i, batch in enumerate(get_batch_sample(100, train_data)):
#             target = [[p[0]] for p in batch]
#             context = [[q[1]] for q in batch]
#             negative = get_negative_sample(centers=target, targets=content, un_table=uni_table, dict=item2index, k=10)
#
#             target = Variable(torch.LongTensor(target))
#             # print(target)
#             context = Variable(torch.LongTensor(context))
#             # print(context)
#             negative = Variable(torch.LongTensor(negative))
#
#             item2vec.zero_grad()
#
#             loss = item2vec(target, context, negative)
#             loss.backward()
#             optimizer.step()
#
#             print('Epoch : %d, Batch : %d, loss : %.04f' % (epoch + 1, i + 1, loss))
#
# item2vec.eval()
#
# print(item2vec.prediction(torch.LongTensor([1])))
# print(item2vec.prediction(torch.LongTensor([1])).data.numpy().tolist())
# print(item2vec.prediction(torch.LongTensor([1])).data.size())
#
# item_embeddings = []
#
# for item, index in item2index.items():
#     print(item)
#     print(index)
#     print(torch.flatten(item2vec.prediction(torch.LongTensor([index]))).data)
#
#     item_embedding_str = str(item) + ";" + ",".join([str(a) for a in torch.flatten(item2vec.prediction(torch.LongTensor([index]))).data.numpy().tolist()])
#
#     item_embeddings.append(item_embedding_str)
#
# with open(r'em.txt', 'w', encoding='utf-8') as g:
#     for st in item_embeddings:
#         g.write(st + "\n")
#
