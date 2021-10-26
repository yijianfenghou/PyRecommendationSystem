import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
import os
import tensorflow as tf

# movieID = str(165108400)
# rank = model.wv.most_similar(movieID, topn=10)
# print(rank)
# workers = multiprocessing.cpu_count(),

class FMModel(tf.keras.Model):

    def __init__(self, num_feat, num_field, reg_l1=0.01, reg_l2=0.01, embedding_size=128):
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2        # l1/l2正则化
        self.num_feat = num_feat    # denote as M
        self.num_field = num_field  # denote as F
        self.embedding_size = embedding_size
        super(FMModel, self).__init__()

        # first order term paramters embedding
        self.first_weights = tf.keras.layers.Embedding(num_feat, 1, embeddings_initializer='uniform')
        self.bias = tf.Variable([0.0], dtype=tf.float32)

        self.feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size, embeddings_initializer='uniform')

    def call(self, feat_index, feat_value):
        # Step1: 先计算得到线性的那一部分
        feat_value = tf.expand_dims(feat_index, axis=-1)
        first_weights = self.first_weights(feat_index)
        first_weight_value = tf.math.multiply(first_weights, feat_value)
        first_weight_value = tf.squeeze(first_weight_value, axis=-1)
        y_first_order = tf.math.reduce_sum(first_weight_value, axis=1)

        # Step2: 计算二阶部分
        second_feat_emb = self.feat_embeddings(feat_index)
        feat_emb_value = tf.math.multiply(second_feat_emb, feat_value)

        # sum_square part
        summed_feat_emb = tf.math.reduce_sum(feat_emb_value, axis=1)
        sum_square_part = tf.math.square(summed_feat_emb)

        # square_sum part
        squared_feat_emb = tf.math.square(feat_emb_value)
        squared_sum_part = tf.math.reduce_sum(squared_feat_emb, axis=1)

        y_second_order = 0.5*tf.subtract(sum_square_part, squared_sum_part)
        output = y_first_order + y_second_order + self.bias
        output = tf.expand_dims(output, axis=1)
        return output


def SkipGram(input_file):
    pullDF = pd.read_csv(input_file, header=1, names=['emp_id', 'operate_time', 'cust_id'])
    pullDF = pullDF.sort_values(by=['emp_id', 'operate_time'])
    custList = pullDF.groupby('emp_id')['cust_id'].apply(list)
    model = Word2Vec(custList, vector_size=32, window=5, sg=1, hs=0, min_count=1, epochs=10, negative=20)

    model.save("./model/Word2vec.w2v")


def readSaveModelFile(save_path):
    w2v = Word2Vec.load("./model/Word2vec.w2v").wv
    cust_ids = w2v.key_to_index
    cust_vectors = w2v.get_normed_vectors()
    cust = dict(zip(cust_ids, cust_vectors))
    print(w2v.index_to_key)


if __name__ == "__main__":
    input_file = "./data/t.csv"
    SkipGram(input_file)

    # save_path = "./model/Word2vec.w2v"
    # readSaveModelFile(save_path)
    #
    # # 训练FM
    # train_batch_dataset =
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    #
    # output = model(idx, value)
    # loss = cross_entropy_loss(y_true=label, y_pred=output)
    #
    # reg_loss = []


