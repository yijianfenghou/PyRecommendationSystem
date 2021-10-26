import argparse
import torch
import torch.nn as nn
from DSSM.AdDSSMNet.DataSet.DataPreprocess import DSSMSomeFeaturesDataSet
from torch.utils.data import DataLoader, random_split, RandomSampler
import string
import numpy as np
import random
from DSSM.AdDSSMNet.model import DSSMModel
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # cosine_distance = F.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(output, 2) + label * torch.pow(
            torch.clamp(self.margin - output, min=0.0), 2))
        # loss_contrastive = -(label*torch.log(output + 1e-8) + (1-label)*torch.log(1-output + 1e-8))

        return loss_contrastive


def train(args, train_dataset, model):
    # 设置dataloader
    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    # 训练集
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=4)
    t_total = args.max_steps
    args.num_train_epoches = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    # 设置优化器
    model.to(args.device)

    # 定义损失函数
    criterion = ContrastiveLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # # 正确的样本数
    # correct = 0.0
    counter = []
    loss_history = []
    iteration_number = 0

    # Training
    for epoch in range(int(args.num_train_epoches)):
        # 损失统计
        running_loss = 0.0
        # 总的样本总数
        total = 0
        # 正确的样本数
        correct = 0.0
        # 正样本,错误正样本,负样本,错误负样本
        TP, FP, TN, FN = 0, 0, 0, 0
        # for batch_i, samples in enumerate(zip(dataLoader, picEmbeddingLoader)):
        for batch_i, samples in enumerate(train_dataLoader):
            model.zero_grad()

            uidAndMatchUid = samples
            uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, pic_embedding = uidAndMatchUid
            # age, work_id, height, sex, match_age, match_work_id, match_height, match_sex, label, pic_embedding = uidAndMatchUid
            # uid, age, match_uid, match_age, label, pic_embedding = uidAndMatchUid

            # 图片的embedding的Double类型转化为Float类型
            # pic_embedding = pic_embedding.float()

            # 输入模型需要的格式表示
            input1 = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
            input2 = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)
            # input1 = torch.cat((uid, age), dim=1).to(torch.int64)
            # input2 = torch.cat((match_uid, match_age), dim=1).to(torch.int64)
            # 输出用户侧和被推荐匹配用户向量的相似度
            output = model(input1, input2, pic_embedding)

            # 定义替换的函数
            zero = torch.zeros_like(label)
            one = torch.ones_like(label)
            # 目标需要降维
            # label = label.squeeze(1)
            label = label.float().unsqueeze(1)
            # 计算用户侧和被推荐侧的余弦相似度(可以使用其他相似度函数)
            # cos_sim = torch.sigmoid(F.cosine_similarity(out1, out2, dim=1))
            # 定义损失函数
            loss_contrastive = torch.mean(criterion(output, label))
            # print(loss_contrastive)
            # print("--------------------------")
            # 使用BCELoss损失函数评价
            # loss = criterion(cos_sim, label)
            # print(criterion(cos_sim, label))
            # loss = torch.sum(criterion(cos_sim, label))
            # 统计每个批次样本的数目
            total += label.size(0)
            # 准确率ACC的计算
            predicted = torch.where(output > 0.5, one, zero).unsqueeze(1)
            # correct += predicted.eq(label.view_as(predicted)).sum().item()
            # print(output.tolist())
            # print(label.tolist())

            # FN predict 0 label 1
            FN += ((predicted == 0) & (label == 1)).cpu().sum().item()
            # FP predict 1 label 0
            FP += ((predicted == 1) & (label == 0)).cpu().sum().item()
            # TP predict 和 label 同时为1
            TP += ((predicted == 1) & (label == 1)).cpu().sum().item()
            # TN predict 和 label 同时为0
            TN += ((predicted == 0) & (label == 0)).cpu().sum().item()

            # 清空梯度
            optimizer.zero_grad()
            # 反向传播，更新参数
            # 针对每个样本损失更新
            # loss.backward(torch.ones_like(loss))
            # 针对minBatch数据进行更新
            # losses.append(loss.item())
            # running_loss += loss
            loss_contrastive.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_i % args.print_interval == 0:
                acc = (TP + TN) / (TP + FP + FN + TN)
                r = TP / (TP + FN + 1)
                # p = TP / (TP + FP)
                # F1 = 2 * r * p / (r + p)
                print(
                    "Epoch number: {} , Current loss: {:.4f}, Current accuracy: {:.4f}%, Current recall: {:.4f}%\n".format(
                        epoch, loss_contrastive.item(), 100. * acc, 100. * r))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
                total = 0
                correct = 0
                TP, FP, TN, FN = 0, 0, 0, 0

    show_plot(counter, loss_history)


def evaluate(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in eval_dataloader:
        inputs, inputs_ids, masks, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss = model(inputs, inputs_ids, masks, labels)
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": float(perplexity)
    }

    return result


def show_plot(iteration, loss):
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--data_path", default="C:/Users/EDZ/Desktop/train_file.xlsx", type=str,
                        help="input data file path")
    parser.add_argument("--charset", default=string.ascii_letters + "-' ", type=str,
                        help="contains all char")
    parser.add_argument("--max_length", default=30, type=int,
                        help="text contains max length")
    parser.add_argument("--hidden_size", default=[128, 64], type=list[int],
                        help="hidden layer size")
    parser.add_argument("--num_epoches", default=20, type=int,
                        help="cycle nums")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="learning rate")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--device", default="cpu", type=str,
                        help="cuda is or not available.")
    parser.add_argument("--output_size", default=32, type=int,
                        help="net output size.")
    parser.add_argument("--pic_embedding_size", default=512, type=int,
                        help="picture features dim.")
    parser.add_argument("--print_interval", default=10, type=int,
                        help="show interval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    dataset = DSSMSomeFeaturesDataSet(args.data_path, args.charset, args.max_length)

    setup_seed(20)

    dataset_size = len(dataset)
    train_size, validation_size = int(0.99999 * dataset_size), dataset_size - int(0.99999 * dataset_size)
    train_set, validation_set = random_split(dataset, [train_size, validation_size])
    # 训练数据集和验证数据集
    train_dataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_dataLoader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    # 输入维度
    input1_size = (
        len(dataset.uid_codec.classes_),
        len(dataset.age_codec.classes_),
        len(dataset.work_id_codec.classes_),
        len(dataset.height_codec.classes_),
        len(dataset.sex_codec.classes_)
    )

    input2_size = (
        len(dataset.uid_codec.classes_),
        len(dataset.age_codec.classes_),
        len(dataset.work_id_codec.classes_),
        len(dataset.height_codec.classes_),
        len(dataset.sex_codec.classes_)
    )

    model = DSSMModel(input1_size, input2_size, args.pic_embedding_size, args.batch_size, args.hidden_size,
                      args.output_size)

# # 参数配置文件
# dataPath = "C:/Users/EDZ/Desktop/train_file.xlsx"
# charset = string.ascii_letters + "-' "
# max_length = 30
# hidden_size = [128, 64]
# batch_size = 128
# num_epoches = 20
# learning_rate = 0.0001
# print_interval = 1

# dataset = DSSMSomeFeaturesDataSet(dataPath, charset, max_length)
# get picture embedding and Prepare model.
# pic_embedding = dataset.pic_embedding
# label = dataset.label

# setup_seed(20)

# dataset_size = len(dataset)
# train_size, validation_size = int(0.99999 * dataset_size), dataset_size - int(0.99999 * dataset_size)
# train_set, validation_set = random_split(dataset, [train_size, validation_size])

# print(validation_set)
# train_set = dataset

# 训练数据集和验证数据集
# train_dataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# validation_dataLoader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# 限定可使用GPU的部分
# device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
# CPU execution
# is_gpu = torch.cuda.is_available()

# input1_size = (
#     len(dataset.uid_codec.classes_),
#     len(dataset.age_codec.classes_),
#     len(dataset.work_id_codec.classes_),
#     len(dataset.height_codec.classes_),
#     len(dataset.sex_codec.classes_)
# )
#
# input2_size = (
#     len(dataset.uid_codec.classes_),
#     len(dataset.age_codec.classes_),
#     len(dataset.work_id_codec.classes_),
#     len(dataset.height_codec.classes_),
#     len(dataset.sex_codec.classes_)
# )
#
# # input1_size = (
# #     len(dataset.uid_codec.classes_),
# #     len(dataset.age_codec.classes_)
# # )
# #
# # input2_size = (
# #     len(dataset.uid_codec.classes_),
# #     len(dataset.age_codec.classes_)
# # )
#
# # output_size = 32
#
# model = DSSMModel(input1_size, input2_size, 512, batch_size, hidden_size, output_size)
#
# print(model)
#
# if is_gpu:
#     model = model.cuda()
#
# # # 优化函数以及损失函数
# # criterion = nn.BCELoss(reduction='none')
# # 定义损失函数
# criterion = ContrastiveLoss()
# # 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
#
# # losses = deque([], maxlen=print_interval)
# # # 正确的样本数
# # correct = 0.0
# counter = []
# loss_history = []
# iteration_number = 0
#
# # Training
# for epoch in range(num_epoches):
#     # 损失统计
#     running_loss = 0.0
#     # 总的样本总数
#     total = 0
#     # 正确的样本数
#     correct = 0.0
#     # 正样本,错误正样本,负样本,错误负样本
#     TP, FP, TN, FN = 0, 0, 0, 0
#     # for batch_i, samples in enumerate(zip(dataLoader, picEmbeddingLoader)):
#     for batch_i, samples in enumerate(train_dataLoader):
#         model.zero_grad()
#
#         uidAndMatchUid = samples
#         uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, pic_embedding = uidAndMatchUid
#         # age, work_id, height, sex, match_age, match_work_id, match_height, match_sex, label, pic_embedding = uidAndMatchUid
#         # uid, age, match_uid, match_age, label, pic_embedding = uidAndMatchUid
#
#         # 图片的embedding的Double类型转化为Float类型
#         # pic_embedding = pic_embedding.float()
#
#         # 输入模型需要的格式表示
#         input1 = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
#         input2 = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)
#         # input1 = torch.cat((uid, age), dim=1).to(torch.int64)
#         # input2 = torch.cat((match_uid, match_age), dim=1).to(torch.int64)
#         # 输出用户侧和被推荐匹配用户向量的相似度
#         output = model(input1, input2, pic_embedding)
#
#         # 定义替换的函数
#         zero = torch.zeros_like(label)
#         one = torch.ones_like(label)
#         # 目标需要降维
#         # label = label.squeeze(1)
#         label = label.float().unsqueeze(1)
#         # 计算用户侧和被推荐侧的余弦相似度(可以使用其他相似度函数)
#         # cos_sim = torch.sigmoid(F.cosine_similarity(out1, out2, dim=1))
#         # 定义损失函数
#         loss_contrastive = torch.mean(criterion(output, label))
#         # print(loss_contrastive)
#         # print("--------------------------")
#         # 使用BCELoss损失函数评价
#         # loss = criterion(cos_sim, label)
#         # print(criterion(cos_sim, label))
#         # loss = torch.sum(criterion(cos_sim, label))
#         # 统计每个批次样本的数目
#         total += label.size(0)
#         # 准确率ACC的计算
#         predicted = torch.where(output > 0.5, one, zero).unsqueeze(1)
#         # correct += predicted.eq(label.view_as(predicted)).sum().item()
#         # print(output.tolist())
#         # print(label.tolist())
#
#         # 各种统计变量
#         # train_correct01 = ((predicted == zero) & (label == one)).cpu().sum().item()
#         # train_correct10 = ((predicted == one) & (label == zero)).cpu().sum().item()
#         # train_correct11 = ((predicted == one) & (label == one)).cpu().sum().item()
#         # train_correct00 = ((predicted == zero) & (label == zero)).cpu().sum().item()
#
#         # FN predict 0 label 1
#         FN += ((predicted == 0) & (label == 1)).cpu().sum().item()
#         # FP predict 1 label 0
#         FP += ((predicted == 1) & (label == 0)).cpu().sum().item()
#         # TP predict 和 label 同时为1
#         TP += ((predicted == 1) & (label == 1)).cpu().sum().item()
#         # TN predict 和 label 同时为0
#         TN += ((predicted == 0) & (label == 0)).cpu().sum().item()
#
#         # 清空梯度
#         optimizer.zero_grad()
#         # 反向传播，更新参数
#         # 针对每个样本损失更新
#         # loss.backward(torch.ones_like(loss))
#         # 针对minBatch数据进行更新
#         # losses.append(loss.item())
#         # running_loss += loss
#         loss_contrastive.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#
#         if batch_i % print_interval == 0:
#             acc = (TP + TN) / (TP + FP + FN + TN)
#             r = TP / (TP + FN + 1)
#             # p = TP / (TP + FP)
#             # F1 = 2 * r * p / (r + p)
#             print(
#                 "Epoch number: {} , Current loss: {:.4f}, Current accuracy: {:.4f}%, Current recall: {:.4f}%\n".format(
#                     epoch, loss_contrastive.item(), 100. * acc, 100. * r))
#             iteration_number += 10
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
#             total = 0
#             correct = 0
#             TP, FP, TN, FN = 0, 0, 0, 0
#
# show_plot(counter, loss_history)
#
# model.eval()

# with torch.no_grad():
#     correct = 0
#     total = 0
#     FN, FP, TP, TN = 0., 0., 0., 0.
#     for validation_data in validation_dataLoader:
#         uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label, pic_embedding = validation_data
#         # 目标需要降维
#         label = label.float().unsqueeze(1)
#         # 输入模型需要的格式表示
#         input1 = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
#         input2 = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)
#         # 输出用户侧和被推荐匹配用户向量的相似度
#         output = model(input1, input2, pic_embedding)
#         # 定义替换的函数
#         zero = torch.zeros_like(label)
#         one = torch.ones_like(label)
#         # 准确率ACC的计算
#         predicted = torch.where(output >= 0.5, one, zero)
#         # 各种统计变量
#         train_correct01 = ((predicted == zero) & (label == one)).cpu().sum().item()
#         train_correct10 = ((predicted == one) & (label == zero)).cpu().sum().item()
#         train_correct11 = ((predicted == one) & (label == one)).cpu().sum().item()
#         train_correct00 = ((predicted == zero) & (label == zero)).cpu().sum().item()
#
#         FN += train_correct01
#         FP += train_correct10
#         TP += train_correct11
#         TN += train_correct00
#
#     print('Test accuracy of the model on the {} test dataset: {} %'.format(len(validation_set), 100 * (TP + TN) / (FN + FP + TP + TN)))
