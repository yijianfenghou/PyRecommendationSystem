import numpy as np
import argparse
import time
import tensorflow as tf

import sys, os
path = os.path.dirname(os.getcwd())
sys.path.insert(0, path)
import pickle

from EGES import *


@tf.function
def fina_model(x, y):
    y_pred = eges(x)
    # lossFunc = SampledNegativeLoss(args.n_sampled, len(side_info), args.embedding_dim)(y_pred, batch_labels)
    lossFunc = SampledNegativeLoss(args.n_sampled, len(side_info), args.embedding_dim)

    loss = lossFunc(y_pred, y)
    model = tf.keras.Model(inputs=[inputs, labels], outputs=[y_pred, loss])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ns_sampled", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str, default='./data_cache/')
    parser.add_argument("--feature_num", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=6)
    parser.add_argument("--outputEmbedFile", type=str, default='./embedding/EGES.embed')
    args = parser.parse_args()

    # read train data
    print('read feature...')
    start_time = time.time()
    side_info = np.loadtxt(args.root_path + 'cust_side_info.csv', dtype=np.int32, delimiter='\t')
    feature_size_list = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_size_list.append(tmp_len)

    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))

    # read data_pair by tf.dataset
    def decode_data_pair(line):
        columns = tf.compat.v1.string_split([line], ' ')
        cust_id = tf.compat.v1.string_to_number(columns.values[0], out_type=tf.int32)
        income = tf.compat.v1.string_to_number(columns.values[1], out_type=tf.int32)
        education = tf.compat.v1.string_to_number(columns.values[2], out_type=tf.int32)
        age = tf.compat.v1.string_to_number(columns.values[3], out_type=tf.int32)
        y = tf.compat.v1.string_to_number(columns.values[4], out_type=tf.int32)
        return tf.stack([cust_id, income, education, age]), y

    dataset = tf.data.TextLineDataset(args.root_path+"all_pairs").map(decode_data_pair, num_parallel_calls=10).prefetch(500000)
    dataset = dataset.repeat(args.epochs)
    dataset = dataset.batch(args.batch_size)
    # iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)   # dataset.make_one_shot_iterator()
    batch_index, batch_labels = iterator.get_next()

    # print("--------------------------")
    # print(batch_index)

    # 训练数据集定义
    inputs = tf.keras.layers.Input(shape=(None,), name="input", dtype=tf.int32)
    labels = tf.keras.layers.Input(shape=(1,), name="label")

    print('read embedding......')
    start_time = time.time()
    model = EGES(feature_size_list, args.feature_num, embedding_dim=args.embedding_dim, lr=args.lr)
    end_time = time.time()
    print('time consumed for read embedding: %.2f' % (end_time - start_time))
    # y_pred = eges(batch_index)
    # # y_pred = eges(inputs)
    #
    # # lossFunc = SampledNegativeLoss(args.n_sampled, len(side_info), args.embedding_dim)(y_pred, batch_labels)
    # lossFunc = SampledNegativeLoss(args.n_sampled, len(side_info), args.embedding_dim)
    #
    # loss = lossFunc(y_pred, labels)
    #
    # model = tf.keras.Model(inputs=[inputs, labels], outputs=[y_pred, loss])
    # model.compile(optimizer="adam")
    # model.fit(batch_index, batch_labels)

    # 开始训练
    for ep in range(args.epochs):
        loss = 0
        for i, x in enumerate(batch_index):
            x = tf.expand_dims(x, axis=0)
            y = batch_labels[i]
            # loss += model.train(np.array([[1,2,3,4]]), labels[i], 1, negative_sampling_num=args.ns_sampled)
            loss += model.train(x.numpy(), y.numpy(), 2, negative_sampling_num=args.ns_sampled)
        print('epoch: '+ str(ep), ' loss: '+ str(loss.numpy()))

    # 得到用户embedding
    usrs = np.array(sorted(set(side_info[:, 0]), key=lambda x:x[0]))
    h = np.array(model.get_embedding(usrs))

    # 保存side information的embedding
    side_info = ['income', 'education', 'age']
    for i in range(1, len(model.cat_embedding)):
        with open(side_info[i] + '_emb.pkl', 'wb') as f:
            pickle.dump(model.cat_embedding[i], f)

    # 提取embedding
    embeddings = {}
    for i, h_i in enumerate(h):
        embeddings[i] = h_i

    with open('embedding.pkl', 'wb') as f:
        pk.dump(embeddings, f)


    # variables = [w for w in eges.cat_embedding] + [eges.alpha_embedding, lossFunc.softmax_w, lossFunc.softmax_b]
    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    #
    # loss_sum = 0
    # for ep in range(2):
    #     with tf.GradientTape() as tape:
    #         loss = lossFunc(y_pred, batch_labels)
    #         grads = tape.gradient(loss, variables)
    #         optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    #         loss_sum += loss.numpy()
    #         print('epoch%s:' % {ep}, loss.numpy())
    #
    # start_time = time.time()





