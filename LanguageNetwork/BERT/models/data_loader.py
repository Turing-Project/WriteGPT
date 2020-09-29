import gc
import glob
import torch
import random

from utils.logging import logger


class Batch(object):

    def __init__(self, data=None, device=None, is_test=False, vy_predict=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)

            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_segments = [x[2] for x in data]
            pre_classes = [x[3] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segments, 0))
            mask = torch.logical_not(src == 0)

            clss = torch.tensor(self._pad(pre_classes, -1))
            mask_cls = torch.logical_not(clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))

            if vy_predict and is_test:
                src_str = [x[-3] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-2] for x in data]
                setattr(self, 'tgt_str', tgt_str)
                doc_id = [x[-1] for x in data]
                setattr(self, 'doc_id', doc_id)
            elif is_test:
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

    @staticmethod
    def _pad(data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    mini_batch, size_so_far = [], 0
    # print(size_so_far)
    for ex in data:
        mini_batch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(mini_batch))
        if size_so_far == batch_size:
            yield mini_batch
            mini_batch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield mini_batch[:-1]
            mini_batch, size_so_far = mini_batch[-1:], simple_batch_size_fn(ex, 1)
    if mini_batch:
        yield mini_batch


def _lazy_dataset_loader(pt_file, corpus_type):
    # print(pt_file)
    dataset = torch.load(pt_file)
    logger.info('Loading %s dataset from %s, number of examples: %d' % (corpus_type, pt_file, len(dataset)))
    # print(dataset)
    return dataset


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        args:
        corpus_type: 'train' or 'valid'
        shuffle:
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]
    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + args.data_name + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + args.data_name + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class DataLoader(object):
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for _batch in self.cur_iter:
                yield _batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset, batch_size=self.batch_size, device=self.device,
                            shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=False):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        # for item in self.dataset:
        #     print(item)
        #     print(item['src'])
        #     exit()
        # print(len(self.dataset))
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def pre_process(self, ex, is_test):
        src = ex['src']
        if 'labels' in ex:
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        if is_test and self.args.vy_predict:
            doc_ic = ex['doc_id']
            return src, labels, segs, clss, src_txt, tgt_txt, doc_ic
        elif is_test:
            return src, labels, segs, clss, src_txt, tgt_txt
        else:
            return src, labels, segs, clss

    def batch_buffer(self, data, batch_size):
        mini_batch, size_so_far = [], 0
        for ex in data:
            if len(ex['src']) == 0:
                continue
            ex = self.pre_process(ex, self.is_test)
            if ex is None:
                continue
            mini_batch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(mini_batch))
            if size_so_far == batch_size:
                yield mini_batch
                mini_batch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield mini_batch[:-1]
                mini_batch, size_so_far = mini_batch[-1:], simple_batch_size_fn(ex, 1)
        if mini_batch:
            yield mini_batch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):
            # print(len(buffer))
            p_batch = sorted(buffer, key=lambda x: len(x[3]))  # size 357
            p_batch = batch(p_batch, self.batch_size)
            p_batch = list(p_batch)  # size 51
            # print(len(p_batch))

            # print(p_batch)
            # print(len(p_batch))
            # exit()

            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, mini_batch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                # print(len(mini_batch))
                _batch = Batch(mini_batch, self.device, self.is_test, vy_predict=self.args.vy_predict)

                yield _batch
            return
