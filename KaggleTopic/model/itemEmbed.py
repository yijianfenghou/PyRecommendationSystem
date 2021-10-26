import os,sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)


def embedding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    item_idx_2_rawid_dict = dict(zip(item_emb_df, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns]
