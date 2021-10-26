from DSSM.AdDSSMNet_tf.DataSet.GoogleBatchData import *
import time
import os
import numpy as np
from DSSM.AdDSSMNet_tf.metric.topk import *

UID_FEATURE_COLUMN = ["uid", "age", "workid", "height", "sex", "watched_history_uid"]
ITEM_FEATURE_COLUMN = ["tuid", "tage", "tworkid", "theight", "tsex", "label"]


def parse_example_data(example):
    '''解析每个样本的example'''
    # 定义解析格式，parse_single_example
    features = {
        'uid': tf.io.FixedLenFeature([], tf.int64),
        'age': tf.io.FixedLenFeature([], tf.int64),
        'workid': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.string),
        'timeStamp': tf.io.FixedLenFeature([], tf.string),
        'watched_history_uid': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'tuid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'tage': tf.io.FixedLenFeature([], tf.int64),
        'tworkid': tf.io.FixedLenFeature([], tf.int64),
        'theight': tf.io.FixedLenFeature([], tf.int64),
        'tsex': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    label_feature = tf.io.parse_single_example(example, features)
    label = tf.cast(label_feature['label'], tf.int32)
    # 若干特征进行名称构造字典
    uid_data = [label_feature['uid'], label_feature['age'], label_feature['workid'], label_feature['height'], label_feature['sex'], label_feature['watched_history_uid']]
    item_data = [label_feature['tuid'], label_feature['tage'], label_feature['tworkid'], label_feature['theight'], label_feature['tsex']]
    uid_feature_dict = dict(zip(UID_FEATURE_COLUMN, uid_data))
    item_feature_dict = dict(zip(ITEM_FEATURE_COLUMN, item_data))
    return uid_feature_dict, item_feature_dict, label


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
    """Reward correction batch """
    return -tf.reduce_mean(reward * tf.math.log(output))


def sampling_p_estimation_single_hash(array_a, array_b, hash_indexs, global_step, alpha=0.01):
    """单Hash函数采样概率估计"""
    array_b[hash_indexs] = (1 - alpha) * array_b[hash_indexs] + \
                           alpha * (global_step - array_a[hash_indexs])
    array_a[hash_indexs] = global_step
    sampling_p = 1 / array_b[hash_indexs]
    return array_a, array_b, sampling_p


def hash_simple(cand_ids, ids_hash_bucket_size):
    return tf.math.mod(cand_ids, ids_hash_bucket_size)


def train_model(strategy, dataset, steps, epochs, ids_column, ids_hash_bucket_size, checkpoints_dir, streaming=False,
                beta=100, lr=0.001):
    """自定义训练"""
    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        left_model, right_model = build_model()

        def pred(left_x, right_x, sampling_p):
            left_y = left_model(left_x, training=True)
            right_y = right_model(right_x, training=True)
            output = corrected_batch_softmax(left_y, right_y, sampling_p=sampling_p)
            return output

        def loss(left_x, right_x, sampling_p, reward):
            output = pred(left_x, right_x, sampling_p)
            return reward_cross_entropy(reward, output)

        def grad(left_x, right_x, sampling_p, reward):
            with tf.GradientTape(persistent=True) as tape:
                loss_value = loss(left_x, right_x, sampling_p, reward)
            left_grads = tape.gradient(loss_value, left_model.trainable_variables)
            right_grads = tape.gradient(loss_value, right_model.train_variables)
            return loss_value, left_grads, right_grads

        epoch_recall_avg = tf.keras.metrics.Mean()
        epoch_positive_avg = tf.keras.metrics.Mean()

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        left_checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=left_model)
        right_checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=right_model)

        left_checkpoint_prefix = os.path.join(checkpoints_dir, "left-ckpt")
        right_checkpoint_prefix = os.path.join(checkpoints_dir, "right-ckpt")

        def train_step(inputs, sampling_p):
            left_x, right_x, reward = inputs
            loss_value, left_grads, right_grads = grad(left_x, right_x, sampling_p, reward)
            optimizer.apply_gradients(zip(left_grads, left_model.trainable_variables))
            optimizer.apply_gradients(zip(right_grads, right_model.trainable_variables))

            epoch_recall_avg.update_state(topk_recall(pred(left_x, right_x, sampling_p), reward))
            epoch_positive_avg.update_state(topk_positive(pred(left_x, right_x, sampling_p), reward))

            return loss_value

        @tf.function
        def distributed_train_step(inputs, sampling_p=None):
            per_replica_losses = strategy.run(train_step, args=(inputs, sampling_p,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        loss_results = []
        recall_results = []
        positive_results = []

        # if tensorboard_dir is not None:
        #     summary_writer = tf.summary.create_file_writer(tensorboard_dir)

        print("Start Traning ... ")
        for epoch in range(epochs):
            if streaming is True:
                array_a = np.zeros(shape=(ids_hash_bucket_size,), dtype=np.float32)
                array_b = np.ones(shape=(ids_hash_bucket_size,), dtype=np.float32) * beta
            total_loss = 0.0
            batches_train_time = 0.0
            batches_load_data_time = 0.0
            epoch_train_time = 0.0
            epoch_load_data_time = 0.0
            step = 1

            batch_load_data_start = time.time()
            for inputs in dataset:
                batch_load_data_stop = time.time()
                if streaming is True:
                    cand_ids = inputs[1].get(ids_column)
                    cand_hash_indexs = hash_simple(cand_ids, ids_hash_bucket_size)
                    array_a, array_b, sampling_p = sampling_p_estimation_single_hash(array_a, array_b, cand_hash_indexs,
                                                                                     step)
                else:
                    sampling_p = None

                batch_train_start = time.time()
                total_loss += distributed_train_step(inputs, sampling_p=sampling_p)
                # total_loss += train_step(inputs, sampling_p=sampling_p)
                batch_train_stop = time.time()
                batch_train_time = batch_train_stop - batch_train_start

                batches_train_time += batch_train_time
                epoch_train_time += batch_train_time

                batch_load_data_time = batch_load_data_stop - batch_load_data_start
                batches_load_data_time += batch_load_data_time
                epoch_load_data_time += batch_load_data_time

                if step % 50 == 0:
                    print("Epoch[{}/{}]: Batch({}/{}) "
                          "DataSpeed: {:.4f}sec/batch "
                          "TrainSpeed: {:.4f}sec/batch "
                          "correct_sfx_loss={:.4f} "
                          "topk_recall={:.4f} "
                          "topk_positive={:.4f}".format(
                        epoch + 1, epochs, step, steps,
                        batches_load_data_time / 50,
                        batches_train_time / 50,
                        total_loss / step,
                        epoch_recall_avg.result(),
                        epoch_positive_avg.result()))
                    batches_train_time = 0.0
                    batches_load_data_time = 0.0
                step += 1
                batch_load_data_start = time.time()

            # optimizer.lr = 0.1 * optimizer.lr

            loss_results.append(total_loss / steps)
            recall_results.append(epoch_recall_avg.result())
            positive_results.append(epoch_positive_avg.result())

            print("Epoch[{}/{}]: correct_sfx_loss={:.4f} topk_recall={:.4f} topk_positive={:.4f}".format(
                epoch + 1, epochs, total_loss / step, epoch_recall_avg.result(), epoch_positive_avg.result()))
            print("Epoch[{}/{}]: Train time: {:.4f}".format(epoch + 1, epochs, epoch_train_time))
            print("Epoch[{}/{}]: Load data time: {:.4f}".format(epoch + 1, epochs, epoch_load_data_time))

            # if tensorboard_dir is not None:
            #     with summary_writer.as_default():  # pylint: disable=not-context-manager
            #         tf.summary.scalar('correct_sfx_loss', total_loss / steps, step=epoch)
            #         tf.summary.scalar('topk_recall', epoch_recall_avg.result(), step=epoch)
            #         tf.summary.scalar('topk_positive', epoch_positive_avg.result(), step=epoch)

            if (epoch + 1) % 2 == 0:
                left_checkpointer.save(left_checkpoint_prefix)
                print(f'Saved checkpoints to: {left_checkpoint_prefix}')
                right_checkpointer.save(right_checkpoint_prefix)
                print(f'Saved checkpoints to: {right_checkpoint_prefix}')

            epoch_recall_avg.reset_states()
            epoch_positive_avg.reset_states()

    return left_model, right_model


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    # 文件路径
    path1 = "C:/Users/EDZ/Desktop/dataset/*"
    # path1 = "path/example.tfrecord/*.tfrecord"

    # filenames = [("/dbfs" + path + "/" + name) for name in os.listdir("/dbfs" + path) if name.startswith("part")]
    # 读取tfrecord数据
    # tf.io.gfile.glob
    # map解析
    dataset = dataset.map(parse_example_data)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # batch_size = 1024
    # dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    #
    # dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓存区中随机打乱数据
    # dataset = dataset.repeat()
    # # dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=pad_shapes,
    # #                                padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
    # # Dataset.prefetch() 开启预加载数据，使得在 GPU 训练的同时 CPU 可以准备数据
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
