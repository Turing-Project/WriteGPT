# encoding=utf-8


import argparse
import time

from utils.logging import init_logger
from utils.prepropress import data_builder
from utils.format_data import split_doc, delete_data, merge_files
from utils.dataio import load_txt_data


def do_format_to_lines(args):
    print(time.process_time())
    data_builder.format_to_lines(args)
    print(time.process_time())


def do_tokenize(args):
    print(time.process_time())
    data_builder.tokenize(args)
    print(time.process_time())


def do_format_to_bert(args):
    print(time.process_time())
    data_builder.format_to_bert(args)
    print(time.process_time())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode", default='greedy', type=str,
                        help='how to generate oracle summaries, `greedy` or `combination`, combination will generate '
                             'more accurate oracles but take much longer time.')
    parser.add_argument("-data_name", default='chinese_summary', help='vy_text')
    parser.add_argument("-oov_test", default=False)
    parser.add_argument("-raw_path", default='./data/raw_data/corpus.csv', help='./data/raw_data/merged.csv')
    parser.add_argument("-split_path", default='./data/split_data/')
    parser.add_argument("-tokenize_path", default='./data/tokenize_data/')
    parser.add_argument("-json_path", default='./data/json_data/')
    parser.add_argument("-map_path", default='./data/map_data/')
    parser.add_argument("-bert_path", default='./data/predict_data/', help='./data/bert_data/')
    parser.add_argument("-oov_bert_path", default='./data/oov_data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=0, type=int)
    parser.add_argument('-max_nsents', default=150, type=int)
    parser.add_argument('-min_src_ntokens', default=0, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('-log_file', default='./logs/chinese_summary.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=10, type=int)
    parser.add_argument('-vy_predict', default=True, type=bool)

    _args = parser.parse_args()
    init_logger(_args.log_file)
    # eval('data_builder.' + _args.mode + '(args)')

    # _pl = [
    #     "./data/raw_data/eval.csv",
    #     "./data/raw_data/test.csv",
    #     "./data/raw_data/train.csv"
    # ]
    # _mp = './data/raw_data/merged.csv'
    # merge_files(_pl, _mp)
    # exit()

    # Split files
    # split_doc(_args.raw_path, _args.split_path)

    # tokenize
    # do_tokenize(_args)
    # Remove all split files
    # delete_data(_args.split_path)

    # Merge files
    # do_format_to_lines(_args)
    # Remove all tokens files
    # delete_data(_args.tokenize_path)


    # Format to bert data
    # do_format_to_bert(_args)
    # Remove all json files
    # delete_data(_args.json_path)

    from utils.format_data import revers_index, filter_data

    # revers_index('./data/raw_data/oov_test.csv')
    # filter_data('./data/raw_data/oov_test.csv')

    # import json
    # with open('./data/json_data/chinese_summary.train.0.json', encoding='utf-8') as f:
    #     data = f.read()
    #     data = json.loads(data)
    # print(type(data))
    # for item in data[:1]:
    #     # print(item)
    #     # print(type(item))
    #     for sub in item:
    #         print(sub)
    #         print(item[sub])
    #         for ssub in item[sub]:
    #             print(ssub)
    # print(data[0])
    json_data_set = data_builder.tokenize_format_lines(_args)

    data_builder.format2bert(_args, json_data_set)
