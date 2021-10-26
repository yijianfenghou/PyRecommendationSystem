from collections import namedtuple

# 使用具有元组定义特殊标记
SparseFeat = namedtuple('SparseFeat', ['name', 'vocab_size', 'embed_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocab_size', 'embed_dim', 'maxlen'])