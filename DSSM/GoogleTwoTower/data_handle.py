import tensorflow as tf


def sampling_p_estimation_single_hash(array_a, array_b, hash_index, global_step, alpha=0.01):
    '''单hash函数采样概率估计'''
    array_b[hash_index] = (1 - alpha) * array_b[hash_index] + alpha * array_a[hash_index]
    array_a[hash_index] = global_step

    sampling_p = 1 / array_b[hash_index]
    return array_a, array_b, sampling_p


def log_q(x, y, sampling_p=None, temperature=0.05):
    """logQ correction used in sampled softmax model."""
    inner_product = tf.reduce_sum(tf.math.multiply(x, y), axis=-1) / temperature
    if sampling_p is not None:
        return inner_product - tf.math.log(sampling_p)

    return inner_product


def corrected_batch_softmax(x, y, sampling_p=None):
    """logQ correction softmax"""
    correct_inner_product = log_q(x, y, sampling_p=sampling_p)
    return tf.math.exp(correct_inner_product) / tf.math.reduce_sum(tf.math.exp(correct_inner_product))


def reward_cross_entropy(reward, output):
    '''Reward correction batch'''
    return -tf.reduce_mean(reward * tf.math.log(output))
