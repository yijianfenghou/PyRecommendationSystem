import argparse
import torch
import torch.nn as nn
from DSSM.AdDSSMNet.DataSet.DataOneHotPreprocess import DSSMSomeFeaturesDataSet
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from DSSM.AdDSSMNet.OneHotModels import DSSMModel
import matplotlib.pyplot as plt
from typing import List


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
        # loss_contrastive = torch.mean((1 - label) * torch.pow(output, 2) + label * torch.pow(
        #     torch.clamp(self.margin - output, min=0.0), 2))
        # loss_contrastive = -(label*torch.log(output + 1e-10) + (1-label)*torch.log(1-output + 1e-10))
        # print(output)
        loss_contrastive = -torch.mean(label * torch.log(output))

        return loss_contrastive


def cosine(self, input1, input2):
    input1_norm = torch.norm(input1, dim=1, keepdim=True)
    input2_norm = torch.norm(input2, dim=1, keepdim=True)

    cosine = torch.sum(torch.mul(input1_norm, input2_norm), dim=1, keepdim=True) / (input1_norm * input2_norm)
    return cosine


def log_q(x, y, sampling_p=None, temperature=0.05):
    """logQ correction used in sampled softmax model."""
    inner_product = torch.sum(torch.mul(x, y), dim=-1) / temperature
    if sampling_p is not None:
        return inner_product * torch.log(sampling_p)
    return inner_product


def corrected_batch_softmax(x, y, sampling_p=None):
    """logQ correction softmax"""
    normx = torch.norm(x, dim=1, keepdim=True)
    normy = torch.norm(y, dim=1, keepdim=True)
    x = torch.div(x, normx)
    y = torch.div(y, normy)
    corrected_inner_product = log_q(x, y, sampling_p)
    return torch.exp(corrected_inner_product) / torch.sum(torch.exp(corrected_inner_product))


def reward_cross_entropy(reward, output):
    """Reward correction softmax"""
    return -torch.mean(reward*torch.log(output))


def topk_recall(output, reward, k=10):
    """TopK Recall rate"""
    _, indices = torch.topk(output, k=k)

    def _true(reward, indices):
        return torch.nonzero(torch.gather(reward, 0, indices)) / k

    def _false():
        return torch.FloatTensor(0)

    if torch.nonzero(reward) > 0:
        return lambda: _true(reward, indices)
    else:
        return lambda: _false()


def train(args, train_dataLoader, validation_dataLoader, model):
    # 设置优化器
    model.to(args.device)

    # 定义损失函数
    criterion = ContrastiveLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    from DSSM.AdDSSMNet.EarlyStopping import EarlyStopping
    # 初始化early_stopping对象
    # patience = 20 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(args.patience, verbose=True)

    # # 正确的样本数
    # correct = 0.0
    counter = []
    loss_history = []
    iteration_number = 0

    # # 标签one_hot_encode化
    # ones = torch.sparse.torch.eye(args.class_num)

    # Training
    for epoch in range(int(args.num_train_epoches)):
        # 损失统计
        running_loss = 0.0
        # 总的样本总数
        total = 0
        # 正样本,错误正样本,负样本,错误负样本
        TP, FP, TN, FN = 0, 0, 0, 0
        # to track the validation loss as the model trains
        valid_losses = []

        model.train()  # 设置模型为训练模式
        for batch_i, samples in enumerate(train_dataLoader):
            model.zero_grad()

            uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label = samples

            # 输入模型需要的格式表示
            user_data = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
            item_data = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)
            # 输出用户侧和被推荐匹配用户向量的相似度
            out1, out2 = model(user_data, item_data)
            output = corrected_batch_softmax(out1, out2, sampling_p=None)

            # 定义替换的函数
            zero = torch.zeros_like(label)
            one = torch.ones_like(label)
            # 目标需要降维
            # label = label.squeeze(1)
            label = label.float().unsqueeze(1)

            # 定义损失函数
            loss_contrastive = criterion(output, label)
            # 统计每个批次样本的数目
            total += label.size(0)
            # 准确率ACC的计算
            predicted = torch.where(output > 0.5, one, zero).unsqueeze(1)

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
                TP, FP, TN, FN = 0, 0, 0, 0

        model.eval()  # 设置模型为评估/测试模式
        for data in validation_dataLoader:
            # forward pass: compute predicted outputs by passing inputs to the model
            uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label = data

            # 输入模型需要的格式表示
            user_data = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
            item_data = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)
            # 输出用户侧和被推荐匹配用户向量的相似度
            out1, out2 = model(user_data, item_data)
            output = corrected_batch_softmax(out1, out2, sampling_p=None)
            # output = model(user_data, item_data)

            print("-------------------------")
            print(output)
            # 目标需要降维
            label = label.float().unsqueeze(1)
            # calculate the loss
            loss = criterion(output, label)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    show_plot(counter, loss_history)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./output/checkpoint.pt'))

    return model


def cal_roc(score, standard):
    from sklearn.metrics import roc_curve, auc

    # 计算出每个模型预测结果对应的fpr，tpr
    # y表示真实标签
    y = standard.cpu().detach().numpy().flatten()
    # predict表示预测出来的值
    predict = score.cpu().detach().numpy().flatten()
    fpr, tpr, threshold = roc_curve(y, predict)
    AUC = auc(fpr, tpr)
    # 只比较auc前5位数字
    AUC = ('%.5f' % AUC)
    return fpr, tpr, AUC


def plot_roc(fpr_list, tpr_list, auc_list, model_name):
    # 在一张图中画出多个model的roc曲线
    fig = plt.figure()
    legend_list = []
    for i in range(len(model_name)):
        # 先x后y
        plt.plot(fpr_list[i], tpr_list[i])
        legend_list.append(str(model_name[i])+'(auc:'+str(auc_list[i])+')')
    plt.legend(legend_list)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Postive Rate')
    plt.title('ROC curve for RNA-disease model')
    # fig.savefig("ROC.png")
    plt.show()
    return


def test(test_size, test_dataset, model):

    model.eval()
    with torch.no_grad():
        fpr_list, tpr_list, auc_list, epochs = [], [], [], []
        FN, FP, TP, TN = 0., 0., 0., 0.
        pos_size = 0
        for i, test_set in enumerate(test_dataset):
            uid, age, work_id, height, sex, match_uid, match_age, match_work_id, match_height, match_sex, label = test_set
            # 定义替换的函数
            zero = torch.zeros_like(label)
            one = torch.ones_like(label)

            # 目标需要降维
            label = label.float().unsqueeze(1)
            # 输入模型需要的格式表示
            input1 = torch.cat((uid, age, work_id, height, sex), dim=1).to(torch.int64)
            input2 = torch.cat((match_uid, match_age, match_work_id, match_height, match_sex), dim=1).to(torch.int64)

            # 输出用户侧和被推荐匹配用户向量的相似度
            output = model(input1, input2)

            # 准确率ACC的计算
            predicted = torch.where(output >= 0.5, one, zero)

            # FN predict 0 label 1
            FN += ((predicted == 0) & (label == 1)).cpu().sum().item()
            # FP predict 1 label 0
            FP += ((predicted == 1) & (label == 0)).cpu().sum().item()
            # TP predict 和 label 同时为1
            TP += ((predicted == 1) & (label == 1)).cpu().sum().item()
            # TN predict 和 label 同时为0
            TN += ((predicted == 0) & (label == 0)).cpu().sum().item()

            pos_size += (label == 1).cpu().sum().item()

            fpr, tpr, AUC = cal_roc(output, label)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(AUC)
            epochs.append(i)

        plot_roc(fpr_list, tpr_list, auc_list, epochs)

        print('Test dataset size: {}, pos label size: {}, accuracy: {}, recall: {}'.format(test_size, pos_size, (TP + TN) / (FN + FP + TP + TN), TP / (TP + FN)))

        import sys
        sys.exit()


def show_plot(iteration, loss):
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## config parameters
    parser.add_argument("--data_path", default="C:/Users/EDZ/Desktop/result.csv", type=str,
                        help="input data file path")
    parser.add_argument("--hidden_size", default=[128, 64], type=List[int],
                        help="hidden layer size")
    parser.add_argument("--num_epoches", default=10, type=int,
                        help="cycle nums")
    parser.add_argument("--max_embed_size", default=50, type=int,
                        help="field's max embedding size")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="one epoch of batch size")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--device", default="cpu", type=str,
                        help="cuda is or not available.")
    parser.add_argument("--output_size", default=32, type=int,
                        help="net output size.")
    parser.add_argument("--pic_embedding_size", default=512, type=int,
                        help="picture features dim.")
    parser.add_argument("--print_interval", default=10, type=int,
                        help="show interval.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epoches", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--class_num", default=2, type=int,
                        help="classes number.")
    parser.add_argument("--patience", default=20, type=int,
                        help="validation dataset's loss nums.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.no_cuda else "cpu")
    args.device = device

    totalDataSet = DSSMSomeFeaturesDataSet(args.data_path)

    dataset_size = len(totalDataSet)
    train_size, validation_size, test_size = int(0.7 * dataset_size), int(0.2 * dataset_size), int(dataset_size - (0.9 * dataset_size))
    train_set, validation_set, test_set = random_split(totalDataSet, [train_size, validation_size, test_size])
    # 训练数据集和验证数据集
    train_dataLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_dataLoader = DataLoader(validation_set, batch_size=validation_size, shuffle=False)
    test_dataLoader = DataLoader(test_set, batch_size=test_size, shuffle=False)

    model = DSSMModel(totalDataSet.user_field_dims, totalDataSet.item_field_dims, args.max_embed_size, args.hidden_size,
                      args.output_size)

    # model = train(args, train_dataLoader, validation_dataLoader, model)

    model.load_state_dict(torch.load('./output/checkpoint.pt'))

    test(test_size, test_dataLoader, model)

    # print(list(model.named_parameters()))
