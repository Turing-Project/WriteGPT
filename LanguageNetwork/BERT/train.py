#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import os
import time
import glob
import torch
import random
import signal
import argparse

from models.trainer import build_trainer
from models import data_loader, model_builder
from models.pytorch_pretrained_bert.modeling import BertConfig

from utils import distributed
from utils.logging import logger, init_logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MultiRunning(object):
    def __init__(self, args, device_id):
        self.args = args
        self.device_id = device_id

    def multi_card_run(self):
        """ Spawns 1 process per GPU """
        init_logger()

        nb_gpu = self.args.world_size
        mp = torch.multiprocessing.get_context('spawn')

        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)

        # Train with multiprocessing.
        process = []
        for i in range(nb_gpu):
            self.device_id = i
            process.append(mp.Process(target=self.multi_card_train, args=(self.args, self.device_id, error_queue),
                                      daemon=True))
            process[i].start()
            logger.info(" Starting process pid: %d  " % process[i].pid)
            error_handler.add_child(process[i].pid)
        for p in process:
            p.join()

    def multi_card_train(self, error_queue):
        """ run process """
        setattr(self.args, 'gpu_ranks', [int(i) for i in self.args.gpu_ranks])

        try:
            gpu_rank = distributed.multi_init(self.device_id, self.args.world_size, self.args.gpu_ranks)
            print('gpu_rank %d' % gpu_rank)
            if gpu_rank != self.args.gpu_ranks[self.device_id]:
                raise AssertionError("An error occurred in Distributed initialization")
            runner = Running(self.args, self.device_id)
            runner.train()
        except KeyboardInterrupt:
            pass  # killed by parent, do nothing
        except Exception:
            # propagate exception to parent process, keeping original traceback
            import traceback
            error_queue.put((self.args.gpu_ranks[self.device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


class Running(object):
    """Run Model"""

    def __init__(self, args, device_id):
        """
        :param args: parser.parse_args()
        :param device_id: 0 or -1
        """
        self.args = args
        self.device_id = device_id
        self.model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval',
                            'rnn_size']

        self.device = "cpu" if self.args.visible_gpus == '-1' else "cuda"
        logger.info('Device ID %d' % self.device_id)
        logger.info('Device %s' % self.device)
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)

        if self.device_id >= 0:
            torch.cuda.set_device(self.device_id)

        init_logger(args.log_file)

    def baseline(self, cal_lead=False, cal_oracle=False):
        test_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'test', shuffle=False),
                                           self.args.batch_size, self.device, shuffle=False, is_test=True)

        trainer = build_trainer(self.args, self.device_id, None, None)

        if cal_lead:
            trainer.test(test_iter, 0, cal_lead=True)
        elif cal_oracle:
            trainer.test(test_iter, 0, cal_oracle=True)

    def train_iter(self):
        return data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'train', shuffle=True),
                                      self.args.batch_size, self.device, shuffle=True, is_test=False)

    def train(self):
        model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=True)

        if self.args.train_from:
            logger.info('Loading checkpoint from %s' % self.args.train_from)
            checkpoint = torch.load(self.args.train_from, map_location=lambda storage, loc: storage)
            opt = vars(checkpoint['opt'])
            for k in opt.keys():
                if k in self.model_flags:
                    setattr(self.args, k, opt[k])
            model.load_cp(checkpoint)
            optimizer = model_builder.build_optim(self.args, model, checkpoint)
        else:
            optimizer = model_builder.build_optim(self.args, model, None)

        logger.info(model)
        trainer = build_trainer(self.args, self.device_id, model, optimizer)
        trainer.train(self.train_iter, self.args.train_steps)

    def validate(self, step):

        logger.info('Loading checkpoint from %s' % self.args.validate_from)
        checkpoint = torch.load(self.args.validate_from, map_location=lambda storage, loc: storage)

        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                setattr(self.args, k, opt[k])
        print(self.args)

        config = BertConfig.from_json_file(self.args.bert_config_path)
        model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=config)
        model.load_cp(checkpoint)
        model.eval()

        valid_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'valid', shuffle=False),
                                            self.args.batch_size, self.device, shuffle=False, is_test=False)
        trainer = build_trainer(self.args, self.device_id, model, None)
        stats = trainer.validate(valid_iter, step)
        return stats.xent()

    def wait_and_validate(self):
        time_step = 0
        if self.args.test_all:
            cp_files = sorted(glob.glob(os.path.join(self.args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            xent_lst = []
            for i, cp in enumerate(cp_files):
                step = int(cp.split('.')[-2].split('_')[-1])
                xent = self.validate(step=step)
                xent_lst.append((xent, cp))
                max_step = xent_lst.index(min(xent_lst))
                if i - max_step > 10:
                    break
            xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
            logger.info('PPL %s' % str(xent_lst))
            for xent, cp in xent_lst:
                step = int(cp.split('.')[-2].split('_')[-1])
                self.test(step)
        else:
            while True:
                cp_files = sorted(glob.glob(os.path.join(self.args.model_path, 'model_step_*.pt')))
                cp_files.sort(key=os.path.getmtime)
                if cp_files:
                    cp = cp_files[-1]
                    time_of_cp = os.path.getmtime(cp)
                    if not os.path.getsize(cp) > 0:
                        time.sleep(60)
                        continue
                    if time_of_cp > time_step:
                        time_step = time_of_cp
                        step = int(cp.split('.')[-2].split('_')[-1])
                        self.validate(step)
                        self.test(step)

                cp_files = sorted(glob.glob(os.path.join(self.args.model_path, 'model_step_*.pt')))
                cp_files.sort(key=os.path.getmtime)
                if cp_files:
                    cp = cp_files[-1]
                    time_of_cp = os.path.getmtime(cp)
                    if time_of_cp > time_step:
                        continue
                else:
                    time.sleep(300)

    def test(self, step=None):
        if not step:
            try:
                step = int(self.args.test_from.split('.')[-2].split('_')[-1])
            except IndexError:
                step = 0

        logger.info('Loading checkpoint from %s' % self.args.test_from)
        checkpoint = torch.load(self.args.test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                setattr(self.args, k, opt[k])

        config = BertConfig.from_json_file(self.args.bert_config_path)
        model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=config)
        model.load_cp(checkpoint)
        model.eval()
        # logger.info(model)
        trainer = build_trainer(self.args, self.device_id, model, None)
        test_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'test', shuffle=False),
                                           self.args.batch_size, self.device, shuffle=False, is_test=True)
        trainer.test(test_iter, step)

    def gen_features_vector(self, step=None):
        if not step:
            try:
                step = int(self.args.test_from.split('.')[-2].split('_')[-1])
            except IndexError:
                step = 0

        logger.info('Loading checkpoint from %s' % self.args.test_from)
        checkpoint = torch.load(self.args.test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                setattr(self.args, k, opt[k])

        config = BertConfig.from_json_file(self.args.bert_config_path)
        model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=config)
        model.load_cp(checkpoint)
        model.eval()
        # logger.info(model)
        trainer = build_trainer(self.args, self.device_id, model, None)
        test_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'test', shuffle=False),
                                           self.args.batch_size, self.device, shuffle=False, is_test=True)
        trainer.gen_features_vector(test_iter, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='transformer', type=str,
                        choices=['classifier', 'transformer', 'rnn', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'vector'])
    parser.add_argument("-data_name", default='chinese_summary', help='vy_text')
    parser.add_argument("-bert_data_path", default='./data/bert_data/', help='./data/bert_data/')
    parser.add_argument("-model_path", default='./models/models_check_points/')
    parser.add_argument("-result_path", default='./results/')
    parser.add_argument("-temp_dir", default='./temp/')
    parser.add_argument("-bert_pretrained_model_path", default='./models/pytorch_pretrained_bert/bert_pretrain/')
    parser.add_argument("-bert_config_path", default='./models/pytorch_pretrained_bert/bert_pretrain/bert_config.json')

    parser.add_argument("-batch_size", default=1000, type=int)

    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=2048, type=int)
    parser.add_argument("-heads", default=8, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optimizer", default='adam', type=str)
    parser.add_argument("-lr", default=2e-3, type=float, help='learning rate')
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='noam', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5000, type=int)
    parser.add_argument("-accum_count", default=2, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=50000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./logs/project.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='./models/models_check_points/model_step_50000.pt')
    parser.add_argument("-train_from", default='', help='./models/models_check_points/model_step_45000.pt')
    parser.add_argument("-validate_from", default='../models/models_check_points/model_step_50000.pt')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-shuffle_data", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-vy_predict", type=str2bool, nargs='?', const=False, default=True)

    _args = parser.parse_args()

    gpu_ranks: str = str(_args.gpu_ranks)
    _args.gpu_ranks = [int(i) for i in gpu_ranks.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = _args.visible_gpus

    init_logger(_args.log_file)
    _device = "cpu" if _args.visible_gpus == '-1' else "cuda"
    _device_id = 0 if _device == "cuda" else -1

    runner = Running(args=_args, device_id=_device_id)
    multi_runner = MultiRunning(args=_args, device_id=_device_id)
    if _args.world_size > 1:
        multi_runner.multi_card_run()
    elif _args.mode == 'train':
        runner.train()
    elif _args.mode == 'validate':
        runner.wait_and_validate()
    elif _args.mode == 'test':
        runner.test()
    elif _args.mode == 'lead':
        runner.baseline(cal_lead=True)
    elif _args.mode == 'oracle':
        runner.baseline(cal_oracle=True)
    elif _args.mode == 'vector':
        runner.gen_features_vector()
