import tensorflow as tf
from tensorflow.keras.layers import Layer


# 自定损失函数，加权交叉熵损失
class SampledNegativeLoss(tf.keras.losses.Loss):

    def __init__(self, num_sampled, num_classes, dim):
        super(SampledNegativeLoss, self).__init__()
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.dim = dim
        # self.softmax_w = tf.Variable(tf.initializers.truncated_normal((self.num_sampled, self.num_classes), stddev=0.1), name='softmax_w')
        self.softmax_w = tf.Variable(tf.random.truncated_normal((self.num_classes, self.dim), mean=0, stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(self.num_classes), name='softmax_b')

    # def build(self):
    #     self.softmax_w = tf.Variable(tf.random.truncated_normal((self.num_classes, self.dim), mean=0, stddev=0.1), name='softmax_w')
    #     self.softmax_b = tf.Variable(tf.zeros(self.num_classes), name='softmax_b')
    #     super(SampledNegativeLoss, self).build()

    def call(self, y_pred, y):
        y = tf.reshape(y, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=y,
            inputs=y_pred,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes
        ))

        return loss


class EGES(Layer):

    def __init__(self, feature_size_list, feature_num, embedding_dim=128, lr=0.001):
        super(EGES, self).__init__()
        # self.feature_lens = features_lens
        # self.side_info = tf.convert_to_tensor(side_info)
        self.feature_sizes = feature_size_list    # 每个side feature的one-hot长度
        self.feature_num = feature_num            # side information特征数量
        self.embedding_dim = embedding_dim        # 维度
        self.lr = lr
        self.softmax_w = tf.Variable(tf.random.truncated_normal((self.feature_sizes[0], self.embedding_dim), mean=0, stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(self.feature_sizes[0]), name='softmax_b')
        self.cat_embedding = self.embedding_init()   # |V| * d
        self.alpha_embedding = tf.Variable(tf.random.truncated_normal((self.feature_sizes[0], self.feature_num), mean=0, stddev=1)) # |V| * feature_num

    # def build(self, input_shape):
    #     self.cat_embedding = self.embedding_init()   # |V| * d
    #     # self.alpha_embedding = tf.Variable(tf.initializers.random_uniform((self.num_samples, self.feature_num), -1, 1))
    #     self.alpha_embedding = tf.Variable(tf.random.truncated_normal((self.feature_sizes[0], self.feature_num), mean=0, stddev=1))   # |V| * feature_num
    #     super(EGES, self).build(input_shape)

    def embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.feature_num):
            embedding_var = tf.Variable(tf.random.truncated_normal((self.feature_sizes[i], self.embedding_dim), mean=0, stddev=1), name='embedding'+str(i), trainable=True)
            # embedding_var = tf.Variable(tf.initializers.random_normal((self.feature_lens[i], self.embedding_dim), -1, 1), name='embedding'+str(i), trainable=True)
            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    @tf.function
    def predict(self, inputs):
        # batch_index = inputs
        # batch_features = tf.nn.embedding_lookup(self.side_info, batch_index)
        embed_list = []
        for i in range(self.feature_num):
            cat_embed = tf.nn.embedding_lookup(self.cat_embedding[i], inputs[:, i])
            embed_list.append(cat_embed)

        # stack变换, 将分开的side information按照 d*n 汇聚在一起
        # (features_num, batch_size, d) => (batch_size, d, features_num)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(tf.exp(self.alpha_embedding), inputs[:, 0])   # alpha是每个node独有的，由id决定
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)                           # 每个id的 exp(A_id)的和

        merge_emb = tf.reduce_sum(stack_embed*alpha_embed_expand, axis=-1) / alpha_i_sum           # 归一化权重聚合

        # model = tf.keras.models.Model(inputs=inputs, outputs=[merge_emb])
        # model.__setattr__("input", inputs)
        # model.__setattr__("item_embedding", merge_emb)
        # 最后输出的embedding为batch*d
        return merge_emb

    def train(self, xs, ys, epoches, negative_sampling_num=0):
        variables = [w for w in self.cat_embedding] + [self.alpha_embedding, self.softmax_w, self.softmax_b]
        optimizer = tf.keras.optimizers.SGD(self.lr)

        # loss_sum = 0
        for epoch in range(epoches):
            with tf.GradientTape() as tape:
                embedding = self.predict(xs)
                loss = self.loss(embedding, ys, ns_num=negative_sampling_num)
                print(loss)
                print("----------------")
                grads = tape.gradient(loss, variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
                # loss_sum += loss.numpy()
        return loss

    def loss(self, embedding, ys, ns_num=0):

        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=tf.reshape(ys, [-1, 1]),
            inputs=embedding,
            num_sampled=ns_num,
            num_classes=self.feature_sizes[0]
        ))
        return loss

    def get_embedding(self, input):
        a = self.alpha_embedding
        w = self.cat_embedding
        a = tf.exp(a)
        a_sum = tf.expand_dims(tf.reduce_sum(a, axis=1), 1)
        a = a/a_sum

        h = []
        for i in range(len(input)):
            h_i = []
            x = input[i]
            id_index = x[0]

            # aggregate
            for j in range(len(x)):
                w_i = w[j][x[j]]
                a_i = a[id_index][j]
                h_i.append(w_i*a_i)
            h_i = tf.reduce_sum(h_i, axis=0)
            h.append(h_i)

        return h
