import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
from os.path import join as pjoin

import torch
from multiprocessing import Pool
from models.pytorch_pretrained_bert import BertTokenizer

from utils.logging import logger
from utils.utils import clean
from utils.prepropress.utils import _get_word_ngrams
from utils.dataio import load_txt_data
from pyparsing import oneOf
from tqdm import tqdm


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p, 'r', encoding='UTF-8'))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if lower:
            tokens = [t.lower() for t in tokens]
        if tokens[0] == '@highlight':
            flag = True
            continue
        if flag:
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    lang = 'zh'
    abstract = sum(abstract_sent_list, [])
    if lang == 'en':
        abstract = _rouge_clean(' '.join(abstract)).split()
    if lang == 'en':
        sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    elif lang == 'zh':
        sents = [' '.join(s).split() for s in doc_sent_list]
    else:
        raise ValueError

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])

    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []

    _rouge = []
    for i in range(len(sents)):
        if i in selected:
            continue
        c = selected + [i]

        candidates_1 = [evaluated_1grams[idx] for idx in c]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        rouge_score = rouge_1 + rouge_2

        _rouge.append(rouge_score)

    for i in range(summary_size):
        if max(_rouge) == 0:
            continue
        idx = _rouge.index(max(_rouge))
        selected.append(idx)
        _rouge[idx] = 0.0

    return selected


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def pre_process(self, src, tgt, oracle_ids, flag_i):

        if len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for ll in oracle_ids:
            labels[ll] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if len(src) < self.args.min_nsents:
            print(1)
            return None
        if len(labels) == 0:
            print(flag_i)
            print(idxs)
            print(src)
            print(tgt)
            print(oracle_ids)
            print(2)
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.json_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1].split('\\')[-1]
            if not args.oov_test:
                a_lst.append((json_f, args, pjoin(args.bert_path, real_name.replace('json', 'bert.pt'))))
            else:
                a_lst.append((json_f, args, pjoin(args.oov_bert_path, real_name.replace('json', 'bert.pt'))))
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    json_file, args, save_file = params
    # print(params[2])
    if os.path.exists(save_file):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    print('Processing %s' % json_file)
    jobs = json.load(open(json_file, 'r', encoding='UTF-8'))
    # print(jobs)
    print('Load json file success')
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        # print(source)
        if args.oracle_mode == 'greedy':
            oracle_ids = greedy_selection(source, tgt, 3)
        elif args.oracle_mode == 'combination':
            oracle_ids = combination_selection(source, tgt, 3)
        else:
            raise ValueError
        if not oracle_ids:
            continue
        b_data = bert.pre_process(source, tgt, oracle_ids)
        if b_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def tokenize(args):
    stories_dir = os.path.abspath(args.split_path)
    tokenized_stories_dir = os.path.abspath(args.tokenize_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if not s.endswith('story'):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    # command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
    #            '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
    #            'json', '-outputDirectory', tokenized_stories_dir]
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir, '-lang', 'zh']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s "
            "(which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized,
                                                                               stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def format_to_lines(args):
    corpus_mapping = {}
    # for corpus_type in ['valid', 'test', 'train']:
    #     temp = []
    #     for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
    #         temp.append(hashhex(line.strip()))
    #     corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.tokenize_path, '*.json')):
        # real_name = f.split('/')[-1].split('.')[0]
        # if real_name in corpus_mapping['valid']:
        #     valid_files.append(f)
        # elif real_name in corpus_mapping['test']:
        #     test_files.append(f)
        # elif real_name in corpus_mapping['train']:
        #     train_files.append(f)
        import random
        if args.oov_test:
            n = 3
        else:
            n = random.randint(1, 50)
        if n == 1 or n == 2:
            valid_files.append(f)
        elif n == 3:
            test_files.append(f)
        else:
            train_files.append(f)
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = "{:s}.{:s}.{:d}.json".format(args.json_path + args.data_name, corpus_type, p_ct)
                with open(pt_file, 'w', encoding='utf-8') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset, ensure_ascii=False))

                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if len(dataset) > 0:
            pt_file = "{:s}.{:s}.{:d}.json".format(args.json_path + args.data_name, corpus_type, p_ct)
            with open(pt_file, 'w', encoding='utf-8') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}


def tokenize_format_lines(args):
    input_path = args.raw_path
    output_path = args.json_path + args.data_name + '.json'
    input_data = load_txt_data(input_path)
    pun = oneOf(list("。，,；：（）()！？\\—、丨/"))
    json_data_set = []
    for i in tqdm(range(len(input_data))):
        json_dict = {}
        raw = input_data[i].split(',', 4)
        if args.vy_predict:
            abstract = [list('NONE')]
            sentence = pun.split(re.sub("\"", '', raw[4]))
        else:
            abstract = [list(raw[0])]
            sentence = pun.split(raw[1])
        split_sentence = []
        for split_content in sentence:
            split_content = list(split_content)
            if split_content:
                split_sentence.append(split_content)

        json_dict['src'] = split_sentence
        json_dict['tgt'] = abstract
        if args.vy_predict:
            json_dict['doc_id'] = raw[0]
        json_data_set.append(json_dict)
    # with open(output_path, 'w', encoding='utf-8') as save:
    #     save.write(json.dumps(json_data_set, ensure_ascii=False))
    return json_data_set


def format2bert(args, json_data_set):
    datasets = []
    bert = BertData(args)
    for i in tqdm(range(len(json_data_set))):
        source, tgt = json_data_set[i]['src'], json_data_set[i]['tgt']
        summary_size = int(len(source) / 2)

        if summary_size > 5:
            summary_size = 5
        if (len(source) == 1 and len(source[0]) == 1) or not source:
            source = 'CAN NOT PREDICT: "{}"'.format(source)

        doc_id = 0
        if args.vy_predict:
            oracle_ids = [0]
            doc_id = json_data_set[i]['doc_id']
        elif args.oracle_mode == 'greedy':
            oracle_ids = greedy_selection(source, tgt, summary_size)
        elif args.oracle_mode == 'combination':
            oracle_ids = combination_selection(source, tgt, summary_size)
        else:
            raise ValueError
        if not oracle_ids and oracle_ids != [0]:
            print('jump')
            continue
        b_data = bert.pre_process(source, tgt, oracle_ids, i)

        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data

        if args.vy_predict:
            b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt, 'doc_id': doc_id}
        else:
            b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                           'src_txt': src_txt, "tgt_txt": tgt_txt}

        datasets.append(b_data_dict)
    # TODO: data type
    data_type = 'test'
    i = 0
    j = 10000
    k = 0

    while i < len(datasets):
        path = args.bert_path + args.data_name + '.{}.{}.bert.pt'.format(data_type, k)
        torch.save(datasets[i:j], path)
        k += 1
        i = j
        j = j + 10000

        logger.info('Saving to %s' % path)
