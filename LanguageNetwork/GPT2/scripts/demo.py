import sys
import os
import argparse
import json
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from modeling import GroverModel, GroverConfig, sample
from tokenization import *
from formatter import coarse_formatter, immediate_print
##### ignore tf deprecated warning temporarily

#tf.logging.set_verbosity(tf.logging.DEBUG)

#try:
#    from tensorflow.python.util import module_wrapper as deprecation
#except ImportError:
#    from tensorflow.python.util import deprecation_wrapper as deprecation
#deprecation._PER_MODULE_WARNING_LIMIT = 0
#deprecation._PRINT_DEPRECATION_WARNINGS = False
#####

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '-metadata_fn',
    dest='metadata_fn',
    type=str,
    help='Path to a JSONL containing metadata',
)
parser.add_argument(
    '-out_fn',
    dest='out_fn',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-input',
    dest='input',
    type=str,
    help='Text to complete',
)
parser.add_argument(
    '-config_fn',
    dest='config_fn',
    default='configs/mega.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-ckpt_fn',
    dest='ckpt_fn',
    default='../models/mega/model.ckpt',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-target',
    dest='target',
    default='article',
    type=str,
    help='What to generate for each item in metadata_fn. can be article (body), title, etc.',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=1,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds. useful if we want to split up a big file into multiple jobs.',
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
)
parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument(
    '-top_p',
    dest='top_p',
    default=0.95,
    type=float,
    help='p to use for top p sampling. if this isn\'t none, use this for everthing'
)
parser.add_argument(
    '-min_len',
    dest='min_len',
    default=1024,
    type=int,
    help='min length of sample',
)
parser.add_argument(
    '-eos_token',
    dest='eos_token',
    default=102,
    type=int,
    help='eos token id',
)
parser.add_argument(
    '-samples',
    dest='samples',
    default=5,
    type=int,
    help='num_samples',
)

def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        'extraction': printable_text(''.join(tokenizer.convert_ids_to_tokens(output_tokens))),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }

args = parser.parse_args()
proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# print("proj_root_path: ", proj_root_path)
vocab_file_path = os.path.join(proj_root_path, "dataset/tokenization/clue-vocab.txt")
# print("vocab_file_path: ", vocab_file_path)
tokenizer = FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)
news_config = GroverConfig.from_json_file(args.config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))

# This controls the top p for each generation.
top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

with tf.compat.v1.Session(config=tf_config, graph=tf.Graph()) as sess:
    initial_context = tf.compat.v1.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.compat.v1.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.compat.v1.placeholder(tf.int32, [])
    min_len = tf.compat.v1.placeholder(tf.int32, [])
    tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                           do_topk=False)

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, args.ckpt_fn)
    print('模型加载好啦！🍺Bilibili干杯🍺 \n')
    print('现在将你的作文题精简为一个句子，粘贴到这里:⬇️，然后回车')
    print("\n")
    print("**********************************************作文题目**********************************************\n")
    text = input()
    print("\n")
    print("**********************************************作文题目**********************************************\n")
    while text != "":
        for i in range(args.samples):
            print("正在生成第 ", i + 1, " of ", args.samples , "篇文章\n")
            print("......\n")
            print("EssayKiller正在飞速写作中，请稍后......\n")

            line = convert_to_unicode(text)
            bert_tokens = tokenizer.tokenize(line)
            encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
            context_formatted = []
            context_formatted.extend(encoded)
            # Format context end

            gens = []
            gens_raw = []
            gen_probs = []

            for chunk_i in range(num_chunks):
                tokens_out, probs_out = sess.run([tokens, probs],
                                                 feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                            eos_token: args.eos_token, min_len: args.min_len,
                                                            p_for_topp: top_p[chunk_i]})

                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
                    gens.append(extraction['extraction'])

            l = re.findall('.{1,70}', gens[0].replace('[UNK]', '').replace('##', ''))
            # print("EssayKilelr正在飞速排版中，请稍后......\n")
            final_output = coarse_formatter("".join(l))
            immediate_print('排版结束，正在输出......\n', final_output)
            print("\n")
            print("把👆复制到Word或其他编辑器中即可转为标准作文排版\n")
            
        print('对作文不满意？想尝试更多题目？ 你可以继续在这里输入:⬇️')
        print("**********************************************作文题目**********************************************\n")
        text = input()
        print("\n")
        print("**********************************************作文题目**********************************************\n")
   