import tensorflow as tf

def topk_recall(output, reward, k=10):
    """TopK Recall rate"""
    _, indices = tf.math.top_k(output, k=k)

    def _true(reward, indices):
        return tf.math.count_nonzero(tf.gather(reward, indices))/tf.math.count_nonzero

    def _false():
        return tf.constant(0., dtype=tf.float64)

    return tf.cond(tf.math.count_nonzero(reward) > 0, lambda: _true(reward, indices), lambda: _false())


def topk_positive(output, reward, k=10):
    """Topk Positive rate"""
    _, indices = tf.math.top_k(output, k=k)

    def _true(reward, indices):
        return tf.math.count_nonzero(tf.gather(reward, indices))/k

    def _false():
        return tf.constant(0., dtype=tf.float64)

    return tf.cond(tf.math.count_nonzero(reward) > 0, lambda: _true(reward, indices), lambda: _false())