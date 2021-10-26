


def FiBiNET(
        linear_feature_columns,
        dnn_feature_columns,
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        l2_reg_dnn=0,
        seed=1024
    ):
    features = build_input_feature(linear_feature_columns + dnn_feature_columns)

    input_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear', l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)

