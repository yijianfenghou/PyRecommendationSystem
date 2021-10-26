from .inputs import *
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class MMOE(Layer):

    def __init__(self, dnn_feature_columns, num_tasks, tasks, tasks_name, num_experts=4, units_experts=128, task_dnn_units=(32, 32), seed=1024, dnn_activation='relu'):
        super(MMOE, self).__init__()
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
    
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.dnn_feature_columns = dnn_feature_columns
        self.num_tasks = num_tasks
        self.tasks = tasks
        self.tasks_name = tasks_name
        self.num_experts = num_experts
        self.units_experts = units_experts
        self.task_dnn_units = task_dnn_units
        self.seed = seed
        self.dnn_activation = dnn_activation
    
    def build(self, input_shape):
        super(MMOE, self).build(input_shape)

    def call(self, inputs):
        # 特征输入
        features = build_embedding_matrix(self.dnn_feature_columns)
        inputs_list = list(features.values())

        # 构建DNN embedding_dict
        dnn_embedding_dict = build_embedding_dict(self.dnn_feature_columns)
        dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(features, self.dnn_feature_columns, dnn_embedding_dict)

        dnn_input = combined_dnn_input(dnn_sparse_embedding_list, dnn_dense_value_list)

        # MMoELayer
        mmoe_layers = MMoELayer(units_experts=self.units_experts, num_tasks=self.num_tasks, num_experts=self.num_experts, name="mmoe_layer")(dnn_input)

        # 分别处理不同Task Tower
        task_outputs = []
        for task_layer, task, task_name in zip(mmoe_layers, self.tasks, self.tasks_name):
            tower_layer = DNN(hidden_units=self.task_dnn_units, activation='relu', name='tower_{}'.format(task_name))(task_layer)

            # batch_size * 1
            output_layer = tf.keras.layers.Dense(
                units=1, activation=None, use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name='logit_{}'.format(task_name)
            )(tower_layer)
            task_outputs.append(output_layer)

        model = Model(inputs=inputs_list, outputs=task_outputs)

        return model

