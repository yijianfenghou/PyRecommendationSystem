from TCN import TCN


# 配置文件
from TCN.DataPreprocessing.data_generator import *

cuda = True                         # 是否使用GPU
data_path = "./data/penn"           # 文件路径
batch_size = 16                     # 每次训练时批量数据大小
nhid = 600                          # 定义神经网络中每层隐藏层单元数
levels = 4                          # 残差模块数，用来计算通道数
emsize = 600                        # 词嵌入长度
k_size = 3                          # 卷积核大小
dropout = 0.45                      # 网络层中的随机dropout比率
emb_dropout = 0.25                  # 嵌入层中的dropout比率
tied = True                         # 是否让编码器和解码器的权重相同
lr = 0.4                            # 初始学习率
optimization = "SGD"                # 梯度下降法
validseqlen = 40                    # 用来验证序列的长度
seq_len = 80                        # 总序列的长度
log_interval = 100                  # 记录最后结果的间隔
clip = 0.35                         # 梯度截断的设定，-1表示不采用梯度截断
epochs = 10                         # 一共训练多少轮

torch.manual_seed(11)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING:you should probably run with --cuda")
corpus = data_generator(data_path)  #得到语料库
eval_batch_size = 10
train_data = batchify(corpus.train, batch_size, cuda)
print("train_data:", train_data.size())
val_data = batchify(corpus.valid, eval_batch_size, cuda)
print("val_data:", val_data.size())
test_data = batchify(corpus.test, eval_batch_size, cuda)
print("test_data:", test_data.size())
n_words = len(corpus.dictionary)#语料库的大小
print("n_words:", n_words)
num_chans = [nhid] * (levels - 1) + [emsize]
print("num_chans", num_chans)

model = TCN(emsize, n_words, num_chans, dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size, tied_weight=tied)
if cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, optimization)(model.parameters(), lr=lr)


def evaluate(data_source):
    model.eval()
    total_loss = 0.0
    processed_data_size = 0
    for i in range(0, data_source.size(1) - 1, validseqlen):
        if i + seq_len - validseqlen >= data_source.size(1) - 1:
            continue
        data, targets = get_batch(data_source, i, seq_len, evaluation=True)
        output = model(data)
        eff_history = seq_len - validseqlen

        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        final_target = targets[:, eff_history:].contiguous().view(-1)
        loss = criterion(final_output, final_target)

        total_loss += (data.size(1) - eff_history) * loss.data

        processed_data_size += data.size(1) - eff_history
    return total_loss.item() / processed_data_size


# 训练
def train():
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, validseqlen)):
        if i + seq_len - validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, seq_len)
        optimizer.zero_grad()
        output = model(data)

        eff_history = seq_len - validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)

        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.data
        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch{:3d}|{:5d}/{:5d} batches | lr {:02.5f} | ms/batch{:5.5f}|loss{:5.2f} |ppl{:8.2f}".format(epoch,
                                                                                                                  batch_idx,
                                                                                                                  train_data.size(
                                                                                                                      1) // validseqlen,
                                                                                                                  lr,
                                                                                                                  elapsed * 1000 / log_interval,
                                                                                                                  cur_loss,
                                                                                                                  math.exp(
                                                                                                                      cur_loss)))
            total_loss = 0
            start_time = time.time()


import math

best_vloss = 1e8
try:
    all_vloss = []
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        test_loss = evaluate(test_data)
        print("-" * 89)
        print("| end of epoch{:3d}|time:{:5.2f}s|valid loss{:5.2f}|valid ppl{:8.2f}".format(epoch, (
                    time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print("| end of epoch{:3d}|time:{:5.2f}s|test loss{:5.2f}|test ppl{:8.2f}".format(epoch, (
                    time.time() - epoch_start_time), test_loss, math.exp(test_loss)))
        print("-" * 89)
        if val_loss < best_vloss:
            with open("model.pt", "wb") as f:
                print("Save model!\n")
                torch.save(model, f)
            best_vloss = val_loss
        if epoch > 5 and val_loss >= max(all_vloss[-5:]):
            lr = lr / 2
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        all_vloss.append(val_loss)
except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")
with open("model.pt", "rb") as f:
    model = torch.load(f)
test_loss = evaluate(test_data)
print("-" * 89)
print("| End of training |test loss {:5.2f} | test ppl{:8.2f}".format(test_loss, math.exp(test_loss)))
print("-" * 89)