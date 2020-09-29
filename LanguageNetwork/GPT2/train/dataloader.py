# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import tensorflow as tf
import sys


sys.path.append("D:/EssayKille_V1/AutoWritter")
#sys.path.append("D:/EssayKiller/train")

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    #example["input_ids"] = tf.sparse_tensor_to_dense(example["input_ids"])
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t


    return example


def input_fn_builder(input_files,
                     seq_length,
                     is_training,
                     num_cpu_threads=8,
                     evaluate_for_fixed_number_of_steps=True):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.

        d = tf.data.TFRecordDataset(input_files)
        # If we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        if evaluate_for_fixed_number_of_steps:
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        #d = d.apply(
        #    tf.data.experimental.map_and_batch(
        #        lambda record: _decode_record(record, name_to_features),
        #        batch_size=batch_size,
        #        num_parallel_batches=num_cpu_threads,
        #        drop_remainder=True))
        print("the actual lens of data is>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", d)
        return d

    return input_fn


#  ~~~~~~~~~~~~~~ This is for classification / AF ~~~~~~~~~~~~~~~~~~
def classification_convert_examples_to_features(
        examples, max_seq_length, batch_size, encoder, output_file, labels, pad_extra_examples=False,
        chop_from_front_if_needed=True):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    label_map = {label: i for i, label in enumerate(labels)}

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # begin_summary is our [CLS] token
        tokens = example['ids'] + [encoder.begin_summary]

        if len(tokens) > max_seq_length:
            if chop_from_front_if_needed:
                tokens = tokens[-max_seq_length:]
            else:
                tokens = example['ids'][:(max_seq_length-1)] + [encoder.begin_summary]
        elif len(tokens) < max_seq_length:
            tokens.extend([encoder.padding] * (max_seq_length - len(tokens)))

        features = collections.OrderedDict()
        features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tokens))
        features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label_map[example['label']]]))
        features['is_real_example'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    if pad_extra_examples:
        for x in range(len(examples) % batch_size):
            features = collections.OrderedDict()
            features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]*max_seq_length))
            features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
            features['is_real_example'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    writer.close()


def classification_input_fn_builder(input_file, seq_length, is_training,
                                    drop_remainder,
                                    buffer_size=100):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=buffer_size)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
