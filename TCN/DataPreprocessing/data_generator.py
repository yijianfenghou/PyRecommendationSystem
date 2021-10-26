import os
import pickle
import torch
from torch.autograd import Variable


# 数据读入和预处理
def data_generator(data_path):
    corpus = Corpus(data_path)  # 生成train, test, valid的语料库
    pickle.dump(corpus, open(data_path + "/corpus", "wb"))
    # pickle.dump(obj, file)是指将obj保存在文件file中
    # file： 对象保存的文件对象， file必须有write()接口
    return corpus


# 将获得单词赋予索引，将word->index，可以理解为生成索引字典
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))

    def tokenize(self, path):
        """ Tokenize a text file"""
        assert os.path.exists(path)  # 断言存在这个路径，如果不存在这个路径，则返回错误
        # 将word添加到dictionary中
        with open(path, "r") as f:
            tokens = 0  # 统计每个文件中有多少字
            for line in f:
                words = line.split() + ['<EOS>']  # 文件中每行单词分开变成字符列表，每一个列表最后一个元素为"<EOS>"
                tokens += len(words)  # 每行的字符个数相加
                for word in words:  # 将每行字放到字典中，如果字典中这个字存在，就给这个字一个索引，最终结果是将每个文件中所有的字都赋予一个索引
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:  # 将文件找的那个每个汉字转化为一个已知的索引，就是将每个字换成序索引，(上边是生成字典，下边引用字典)
            ids = torch.LongTensor(tokens)  # 比如这个文件有73760个汉字，就生成随机的73760个tensor,比如：将第100个汉字随机用156254表示
            token = 0
            for line in f:
                words = line.split() + ['<EOS>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]  # 将随机数转换成索引，比如：将第100个随机表示的数变成第100个汉字在字典中的索引
                    token += 1
            return ids  # 返回的是每个字在字典中的索引


def batchify(data, batch_size, cuda):  # 返回批量化后的数据
    nbatch = data.size(0) // batch_size  # nbatch的批次次数
    data = data.arrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1)
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, seq_len, seq_le=None, evaluation=False):
    seq_le = min(seq_le if seq_le else seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i + seq_le], volatile=evaluation)
    target = Variable(source[:, i + 1:i + 1 + seq_le])
    return data, target


