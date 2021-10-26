from Utils.utils import *
from Utils.input_column import *
from tensorflow.keras.layers import Dense, Add, Activation
from tensorflow.keras.models import Model


class AFM_layer(Layer):
    def __init__(self, att_dims=8):
        super(AFM_layer, self).__init__()
        self.att_dims = att_dims

    def build(self, input_shape):
        embed_dims = input_shape[0][-1]
        self.att_W = self.add_weight(name='W', shape=(embed_dims, self.att_dims), initializer='glorot_uniform', regularizer='l2', trainable=True)
        self.att_b = self.add_weight(name='b', shape=(self.att_dims,), initializer='zeros', trainable=True)
        self.p_h = self.add_weight(name='h', shape=(self.att_dims, 1), initializer='glorot_uniform', regularizer='l2', trainable=True)
        self.p_p = self.add_weight(name='p', shape=(embed_dims, 1), initializer='glorot_uniform', regularizer='l2', trainable=True)




def AFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input(),这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_feature(linear_feature_columns, dnn_feature_columns)

    # 将linear部分的特征中sparse特征筛选出来，后面用来做1维embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的形式对应，是通过模型的输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    # linear_logits有两部分组成，分别是dense特征的logits和sparse特征的logits
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # embedding层用户构建FM交叉部分和DNN的输入部分
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 将输入的dnn中sparse特征筛选出来
    att_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    att_logits = get_attention_logits(sparse_input_dict, att_sparse_feature_columns, embedding_layers)

    # 将linear, dnn的logits相加作为最终的logits
    output_logits = Add()([linear_logits, att_logits])

    # 这里的激活函数使用sigmoid
    output_layers = Activation("sigmoid")(output_logits)

    model = Model(input_layers, output_layers)
    return model

