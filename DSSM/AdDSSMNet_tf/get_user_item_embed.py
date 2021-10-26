import numpy as np

test_user_model_input = {
    "user_id": np.array(X["user_id"]),
    "gender": np.array(X["gender"]),
    "age": np.array(X["age"]),
    "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X["hist_movie_id"]]), \
    "hist_len": np.array(X["hist_len"])
}
test_item_model_input = {
    "movie_id": np.array(X["movie_id"]),
    "movie_type_id": np.array(X["movie_type_id"])
}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
item_embs = item_embedding_model.predict(test_item_model_input, batch_size=2 ** 12)
