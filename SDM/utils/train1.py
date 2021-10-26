import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras.backend as K
import os, sys
path = os.path.dirname(__file__)
sys.path.append(path)
from preprocess import gen_data_set_sdm, get_model_input_sdm


def main(data_path):
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(data_path + "ml-1m/users.dat", sep="::", header=None, names=unames)
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path + "ml-1m/ratings.dat", sep="::", header=None, names=rnames)
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(data_path + "ml-1m/movies.dat", sep="::", header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)

    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", "genres"]

    SEQ_LEN_short = 5
    SEQ_LEN_prefer = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    train_set, test_set = gen_data_set_sdm(data, seq_short_len=SEQ_LEN_short, seq_prefer_len=SEQ_LEN_prefer)

    train_model_input, train_label = get_model_input_sdm(train_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)
    test_model_input, test_label = get_model_input_sdm(test_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)

    print(train_model_input)
    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    embedding_dim = 32

    user_feature_columns = [
        SparseFeat('user_id', feature_max_idx['user_id'], 16),
        SparseFeat('gender', feature_max_idx['gender'], 16),
        SparseFeat('age', feature_max_idx['age'], 16),
        SparseFeat('occupation', feature_max_idx['occupation'], 16),
        SparseFeat('zip', feature_max_idx['zip'], 16),
        VarLenSparseFeat(SparseFeat('short_movie_id', feature_max_idx['movie_id'], embedding_dim, embedding_name="movie_id"), SEQ_LEN_short, combiner='mean', length_name='short_sess_length'),
        VarLenSparseFeat(SparseFeat('prefer_movie_id', feature_max_idx['movie_id'], embedding_dim, embedding_name="movie_id"), SEQ_LEN_prefer, combiner='mean', length_name='prefer_sess_length'),
        VarLenSparseFeat(SparseFeat('short_genres', feature_max_idx['genres'], embedding_dim, embedding_name='genres'), SEQ_LEN_short, combiner='mean', length_name='short_sess_length'),
        VarLenSparseFeat(SparseFeat('prefer_genres', feature_max_idx['genres'], embedding_dim, embedding_name='genres'), SEQ_LEN_short, combiner='mean', length_name='prefer_sess_length')
    ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    K.set_learning_phase(True)

    model = SDM(user_feature_columns, item_feature_columns, history_feature_list=['movie_id', 'genres'], units=embedding_dim, num_sampled=100,)

    # 梯度裁剪
    optimizer = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)

    # model.compile(optimizer=optimizer, loss=)

if __name__ == "__main__":
    data_path = "../data/"
    main(data_path)