# -*- coding: utf-8 -*-


"""
Turn a merged corpus into tfrecord files.

NOTE: You will want to do this using several processes. I did this on an AWS machine with 72 CPUs using GNU parallel
as that's where I had the deduplicated RealNews dataset.
"""

#python prepare_data.py -input_fn /data/home/share1/gpt2-ml-Finetune/data

import argparse
import ujson as json
# from sample.encoder import get_encoder, tokenize_for_grover_training, detokenize, sliding_window, create_int_feature
import random
import tensorflow as tf
import collections
import os
import sys
from tempfile import TemporaryDirectory
sys.path.append("D:/EssayKiller_V1/")
from tokenization import tokenization


if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(
      ['Essay_Killer_0001.tfrecord'], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'input_ids': tf.FixedLenFeature([1024], tf.int64)
      })

    image = tf.cast(features['input_ids'], tf.int64)

 
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      for i in range(1):
        example = sess.run([image])
        count = 1
        for i in example[0]:
            if i == 0:continue
            count += 1
            #print(i)
        print("total lens: ", count)
        print("unique lens: ", len(set(example[0])))
      coord.request_stop()
      coord.join(threads)