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

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-input_dir',
    dest='input_dir',
    default='./',
    type=str,
    help='which fold we are on'
)
parser.add_argument(
    '-output_dir',
    dest='output_dir',
    default='.\training_data\Essay_Killer',
    type=str,
    help='which dir we output splitted json files',
)
parser.add_argument(
    '-partition',
    dest='partition',
    default=10,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=12060,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=1337,
    type=int,
    help='which seed to use'
)

print("now begin...")
tokenizer = tokenization.FullTokenizer(
    vocab_file="D:/EssayKiller_V1/AutoWritter/dataset/clue-vocab.txt", do_lower_case=True)
    # OK now write the tfrecord file
total_written = 0

print("begin again...")
def json_spliter(input_dir,output_dir,partition, num_folds):
    with open(input_dir, 'r',encoding='utf-8') as f:
        line_count = 0
        tf_count = 0
        result = []
        for jsonstr in f.readlines(): # 按行读取json文件，每行为一个字符串
            data = json.loads(jsonstr) # 将字符串转化为列表
            print("line_count>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", line_count)
            if line_count % partition != 0 and line_count < num_folds or line_count < partition:
                result.append(data)
                line_count += 1
                print("result>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", result)
                continue
            else:
                for line in result:
                    # print("tf_count>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", tf_count)
                    with open(output_dir + '_{:04d}.json'.format(tf_count) , 'a') as f:         
                        json.dump(line, f,ensure_ascii=False) # 将字符串写入新的json文件中（newfile需要提前定义）        
                        f.write('\n') 
                tf_count += 1
                line_count += 1
                result = []
    print("spliting finished.")


def main():

    args = parser.parse_args()
    random.seed(args.seed + args.num_folds)

    print('args:\n' + args.__repr__())
    num_folds = args.num_folds
    input_dir = args.input_dir
    output_dir = args.output_dir
    partition = args.partition
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>0 \n", args)
    json_spliter(input_dir,output_dir, partition, num_folds)


if __name__ == '__main__':
    main()
