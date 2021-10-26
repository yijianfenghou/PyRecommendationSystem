class HParams:
    def __init__(self):
        self.dropout = 0.5
        # onehotencoding之前的维度特征
        self.FIELD_COUNT = 55
        # onehotencoding之后的维度特征
        self.FEATURE_COUNT = 210
        self.dim = 256
        self.layer_sizes = [512, 256, 128]
        self.cross_layer_sizes = [256, 256, 256]