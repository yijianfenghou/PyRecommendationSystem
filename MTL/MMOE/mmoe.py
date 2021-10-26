import tensorflow as tf
import tensorflow.keras.backend as K

# 1. expert_kernels (input_features * hidden_units * num_experts = 4 * 2 * 3)
# input_features = 4
# hidden_units = 2
# num_experts = 3

export_kernels = tf.constant([
    [[1., 1., 1.], [2., 2., 1.]],
    [[0.1, 0.5, 1.], [0.4, 0.1, 1.]],
    [[1., 1., 1.], [2., 2., 1.]],
    [[0., 1., 6.], [0., 2., 0.]]
    ], dtype=tf.float64
)

# 2. gate_kernels (input_features * num_experts * num_tasks = 4 * 3 * 2)
gate_kernels = [tf.constant([[0.1, 0.5, 1.], [0.4, 0.1, 1.], [1., 1., 1.], [2., 2., 1.]], dtype=tf.float64),
                tf.constant([[1., 2., 1.], [4., 0.2, 1.5], [2., 1., 0.], [5., 2., 1.]], dtype=tf.float64)]

# 3. input samples (samples * input_features = 2 * 4)
# input_features = 4
inputs = tf.constant([[1., 2., 1., 0.], [4., 0.2, 1., 1.]], dtype=tf.float64)

# 4. expert_outputs = input * expert_kernels (samples * hidden_units * num_experts = 2 * 2 * 3)
# f_{i}(x) = activation(W_{i} * x + b)
# samples = 2, hidden_units = 2, num_experts = 3
export_outputs = tf.tensordot(a=inputs, b=export_kernels, axes=1)

# 5. gate_outputs = input * gate_kernels (num_tasks * samples * num_experts = 2 * 2 * 3)
# g^{k}(x) = activation(W_{g,k} * x + b)
gate_outputs = []
for index, gate_kernel in enumerate(gate_kernels):
    gate_output = K.dot(x=inputs, y=gate_kernel)
    gate_outputs.append(gate_output)

gate_outputs = tf.nn.softmax(gate_outputs)

# 6. final_result = gate_outputs * expert_outputs (num_tasks * samples * hidden_units = 2 * 2 * 2)
# 每个 task 的权重值 (gate_output) 分别作用于 expert_outputs，根据 hidden_units 维度进行加和
# f^{k}(x) = sum_{i=1}^{n} (g^{k}_{i}(x) * f_{i}(x))
final_outputs = []
hidden_units = 2

for gate_output in gate_outputs:
    expandad_gate_output = K.expand_dims(gate_output, axis=1)

    weighted_export_output = export_outputs * K.repeat_elements(expandad_gate_output, hidden_units, axis=1)

    final_outputs.append(K.sum(weighted_export_output, axis=2))

print("final_output: \n", final_outputs)
