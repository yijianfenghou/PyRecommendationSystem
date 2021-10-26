import tensorflow as tf


class EGES:

    def __init__(self, feature_num, feature_size_list, nodes_num, learning_rate=0.01, embedding_size=32):
        super(EGES, self).__init__()

        self.feature_num = feature_num          # side information特征数量
        self.feature_sizes = feature_size_list  # 每个side feature的one-hot长度
        self.embedding_size = embedding_size    # 论文中的d
        self.learning_rate = learning_rate

        # eges的模型参数
        self.side_info_weights = self.set_si_vars()   # |V|*d
        self.uid_weights = tf.Variable(tf.random.truncated_normal([self.feature_sizes[0], self.embedding_size], mean=0, stddev=1, dtype=tf.float32), name="uid_weights")

    # 定义side information的embedding matrices
    # tf.random.truncated_normal([self.feature_sizes[i],self.embedding_size], mean=0, stddev=1, dtype=tf.dtypes.float32)
    def set_si_vars(self):
        si_all = []
        for i in range(self.feature_num):
            si_all.append(tf.Variable(tf.random.truncated_normal([self.feature_sizes[i], self.embedding_size], mean=0, stddev=1, dtype=tf.float32), name="side_information_"+str(i)))
        return si_all

    @tf.function
    def predict(self, input):
        embedding_list = []

        # side information embedding
        for i in range(self.feature_num):
            embedding_list.append(tf.nn.embedding_lookup(self.side_info_weights[i], input[:, i]))

        # stack变换, 将分开的side information按照 d*n 汇聚在一起
        # (features_num, batch_size, d) => (batch_size, d, features_num)
        x_embs = tf.stack(embedding_list, axis=-1)

        # weighted aggregation
        exp_A = tf.exp(self.uid_weights)
        x_weight = tf.nn.embedding_lookup(exp_A, input[:, 0])          # A是每个node独有的，由id决定
        x_weight = tf.expand_dims(tf.exp(x_weight), 1)
        x_weight_sum = tf.reduce_sum(x_weight, axis=-1)                # 每个id的exp(A_id)的和
        agg_x = tf.reduce_sum(x_embs*x_weight, axis=-1)/x_weight_sum   # 归一化权重聚合

        # 最后输出的embedding为batch*d
        return agg_x

    def train(self, xs, samples, ys, epoches):
        variables = [w for w in self.side_info_weights] + [self.uid_weights]
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        with tf.GradientTape() as tape:
            loss = self.loss(xs, samples, ys)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    def loss(self, xs, samples, ys):
        h1 = self.predict(xs)
        h2 = self.predict(samples)
        y_ = tf.reduce_sum(tf.multiply(h1, h2), axis=1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(ys, y_)
        return loss

# 方案2，利用tf官方文档的w2v的损失函数， 自动实现了negative sampling
class EGES_V2:

    def __init__(self, feature_num, feature_size_list, learning_rate=0.01, embedding_size=64):
        super(EGES_V2, self).__init__()

        self.feature_num = feature_num            # side information特征数量,
        self.feature_sizes = feature_size_list    # 每个side feature的one-hot长度
        self.embedding_size = embedding_size      # 论文中的d
        self.learning_rate = learning_rate        # lr

        # eges的模型参数
        self.side_ws = self.set_si_vars()         # |V| * d
        self.A = tf.Variable(tf.random.truncated_normal([self.feature_sizes[0], self.feature_num], mean=0, stddev=1, dtype=tf.dtypes.float32), name='A')   # |V| * feature_num
        self.softmax_weights = tf.Variable(tf.random.truncated_normal([self.feature_sizes[0],embedding_size], mean=0, stddev=1, dtype=tf.dtypes.float32), name='soft_w')
        self.softmax_biases = tf.Variable(tf.zeros(self.feature_sizes[0]), name='soft_b')

    # 定义side information的embedding matrices
    def set_si_vars(self):
        si_all = []
        for i in range(self.feature_num):
           si_all.append(tf.Variable(tf.random.truncated_normal([self.feature_sizes[i],self.embedding_size], mean=0, stddev=1, dtype=tf.dtypes.float32), name='side_information_'+str(i)))
        return si_all

    @tf.function
    def predict(self, input):
        embedding_list = []

        # side information embdding
        for i in range(self.feature_num):
            si_embedding = tf.nn.embedding_lookup(self.side_ws[i], input[:, i])
            embedding_list.append(si_embedding)

        # stack变换, 将分开的side information按照 d*n 汇聚在一起
        # (features_num, batch_size, d) => (batch_size, d, features_num)
        x_embs = tf.stack(embedding_list, axis=-1)

        # weighted aggregation
        exp_A = tf.exp(self.A)
        x_weight = tf.nn.embedding_lookup(exp_A, input[:, 0])  # A是每个node独有的，由id决定
        x_weight = tf.expand_dims(tf.exp(x_weight), 1)
        x_weight_sum = tf.reduce_sum(x_weight, axis=-1)  # 每个id的 exp(A_id)的和
        agg_x = tf.reduce_sum(x_embs * x_weight, axis=-1) / x_weight_sum  # 归一化权重聚合

        # 最后输出的embedding为batch*d
        return agg_x

    def train(self, xs, ys, epoches, negative_sampling_num=0):
        variables = [w for w in self.side_ws] + [self.A, self.softmax_weights, self.softmax_biases]
        optimizer = tf.keras.optimizers.SGD(self.learning_rate)

        loss_sum = 0
        for epoch in range(epoches):
            with tf.GradientTape() as tape:
                embedding = self.predict(xs)
                loss = self.loss(embedding, ys, ns_num=negative_sampling_num)
                grads = tape.gradient(loss, variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
                loss_sum += loss.numpy()
                # print('epoch%s:'%{epoch}, loss.numpy())

        return loss

    def loss(self, embedding, ys, ns_num=0):
        loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.softmax_weights,
                        biases=self.softmax_biases,
                        labels=tf.reshape(ys, [-1, 1]),
                        inputs=embedding,
                        num_sampled=ns_num,
                        num_classes=self.feature_sizes[0]
                    )
                )
        return loss

    def get_embedding(self, input):
        a = self.A
        # w = np.array([w.numpy() for w in self.side_ws])

        w = self.side_ws

        a = tf.exp(a)
        a_sum = tf.expand_dims(tf.reduce_sum(a, axis=1), 1)
        a = a / a_sum

        h = []
        for i in range(len(input)):
            h_i = []
            x = input[i]
            id_index = x[0]

            # aggregate
            for j in range(len(x)):
                w_i = w[j][x[j]]
                a_i = a[id_index][j]
                h_i.append(w_i * a_i)
            h_i = tf.reduce_sum(h_i, axis=0)
            h.append(h_i)

        return h

