import os, json, codecs
import tensorflow as tf
import DSSM.AdDSSMNet_tf.utils.config as config

FLAGS = config.FLAGS

def parse_exp(example):
    features_def = dict()
    features_def["label"] = tf.io.FixedLenFeature([1], tf.int64)
    pass