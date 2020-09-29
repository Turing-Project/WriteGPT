from utils.dataio import load_txt_data, save_txt_file, load_file_name, delete_file
from tqdm import tqdm


class DataConverter(object):
    def __init__(self):
        pass

    def add_data(self):
        pass


def merge_files(path_list, merge_path):
    data = []
    for path in path_list:
        data += load_txt_data(path)
    save_txt_file(data, merge_path)


def split_doc(data_path, out_path):
    data = load_txt_data(data_path)
    doc_index = 0
    for i in tqdm(range(len(data)), desc='split_doc'):
        line = data[i].split(',')
        abstract = " ".join(line[0])
        from pyparsing import oneOf
        punc = oneOf(list("。，；；！？"))
        document = [' '.join(x) for x in punc.split(line[1])]
        # print(document)
        for j in range(len(document)):
            document[j] = document[j] + '\n'
        new_doc = document + ['@highlight\n'] + [abstract]
        _doc_index = str(doc_index)
        while len(_doc_index) <= 8:
            _doc_index = '0' + _doc_index
        save_txt_file(new_doc, out_path + _doc_index + '.story')
        doc_index += 1


def split_doc2(data_path, out_path):
    import re
    data = load_txt_data(data_path)
    doc_index = 0
    for i in tqdm(range(len(data))):
        try:
            line = data[i].split(',')
            if len(line[0]) < 100:
                continue
            abstract = re.sub("[\" ]", "", line[1])
            abstract = ' '.join(abstract)
            tmp = re.sub("[\" ]", "", line[0])
            tmp = tmp.split('。')
            document = []
            for x in tmp:
                document.append(' '.join(x))
        except IndexError:
            continue

        # print(document)
        for j in range(len(document)):
            document[j] = document[j] + '\n'
        new_doc = document + ['@highlight\n'] + [abstract]
        save_txt_file(new_doc, out_path + str(doc_index) + '.story')
        doc_index += 1


def delete_data(path, b_range=None, e_range=None):
    all_files = load_file_name(path)[2]
    for i in tqdm(all_files, desc="delete files"):
        if 'ignore' not in i:
            delete_file(path + i)


def revers_index(path):
    data = load_txt_data(path)
    res = []
    for item in data:
        raw = item.split(',')
        doc = raw[0]
        try:
            abst = raw[1]
        except IndexError:
            continue
        res.append('{},{}'.format(abst, doc))
    save_txt_file(res, path)


def filter_data(path):
    data = load_txt_data(path)
    res = []
    for item in tqdm(data, desc='Filter'):
        raw = item.split(',')
        doc = raw[1]
        abst = raw[0]
        if len(doc) >= 100:
            res.append('{},{}'.format(abst, doc))
    save_txt_file(res, path)


if __name__ == '__main__':
    _pl = [
        "./data/eval.csv",
        "./data/test.csv",
        "./data/train.csv"
    ]
    _mp = '../data/raw_data/merged.csv'
    merge_files(_pl, _mp)
    # _op = '../data/raw_data'
    # split_doc(_mp, _op)
    _s = '。'
    # delete_data('./data/split_data/', 100000)
