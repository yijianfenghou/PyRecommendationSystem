import os
import time
import math
import pathlib
import functools
import numpy as np
import tensorflow as tf

from .model import build_model

def parse_csv_line(left_columns,
                   right_columns,
                   csv_header,
                   csv_line):
    '''解析csv数据'''

    def _parse_columns(columns):
        columns_defs = []
        columns_indexs = []
        for col in columns:
            columns_indexs.append(csv_header.index(col))
            if "time" in col or "count" in col:
                columns_defs.append(tf.constant(-1, dtype=tf.int32))
            else:
                columns_defs.append(tf.constant(""))

        parsed_columns = tf.io.decode_csv(
            csv_line,
            record_defaults=columns_defs,
            select_cols=columns_indexs,
            use_quote_delim=False
        )
        return parsed_columns

    def _parse_multi_hot(tensor, max_length=10, duplicate=True, delim='_'):
        """Multi-hot"""
        vals = tensor.numpy().decode('utf-8').strip(delim).split(delim)
        if duplicate is True:
            vals = list(set(vals))
        if len(vals) < max_length:
            vals = vals + ['']*(max_length-len(vals))
        if len(vals) > max_length:
            vals = vals[:max_length]
        return tf.constant(vals)

    left_parsed_columns = _parse_columns(left_columns)
    right_parsed_columns = _parse_columns(right_columns)
    labels = _parse_columns(['label'])

    left_features = dict(zip(left_columns, left_parsed_columns))
    right_features = dict(zip(right_columns, right_parsed_columns))
