import random
import numpy as np
import pandas as pd
import torch
import NCF.src.config as config

class NCF_Data(object):
    """
    Construct Dataset for NCF
    """
    def __init__(self, args, ratings):
        self.ratings = ratings
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size

        self.preprocess_ratings = self._reindex(self.ratings)

        self.user_pool = set(self.ratings["user_id"].unique())
        self.item_pool = set(self.ratings["item_id"].unique())

        self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
        self.negatives = self._negative_sampling(self.preprocess_ratings)
        random.seed(args.seed)

    def _reindex(self, ratings):
        """
        Process dataset to reindex userID and itemID, also set rating as binary feedback
        """
        user_list = list(ratings["user_id"].drop_duplicates())
        user2id = {w: i for i,w in enumerate(user_list)}

        item_list = list(ratings["item_id"].drop_duplicates())
        item2id = {w: i for i,w in enumerate(item_list)}

        ratings["user_id"] = ratings["user_id"].apply(lambda x: user2id[x])
        ratings["item_id"] = ratings["item_id"].apply(lambda x: item2id[x])
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x > 0))

        return ratings

    def _leave_one_out(self, ratings):
        """
        leave-one-out evaluation protocal in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        ratings["rank_latest"] = ratings.groupby(['user_id'])["timestamp"].rank(method='first', ascending=False)
        test = ratings.loc[ratings["rank_latest"] == 1]
        train = ratings.loc[ratings["rank_latest"] > 1]
        assert train['user_id'].nunique() == test["user_id"].nunique(), "Not Match Train User with Test user"
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

    def _negative_sampling(self, ratings):
        interact_status = {
            ratings.groupby('user_id')['item_id'].
            apply(set).
            reset_index().
            rename(columns={'item_id': 'interacted_items'})
        }
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(x, self.num_ng_test))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]

    def get_train_instance(self):
        users, items, rating = [], [], []
        train_ratings = pd.merge()

