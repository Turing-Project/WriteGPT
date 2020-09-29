import torch
import random
from utils.logging import logger, init_logger
from models.pytorch_pretrained_bert.modeling import BertConfig
from models import data_loader, model_builder
from models.trainer import build_trainer
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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

        try:
            self.step = int(self.args.test_from.split('.')[-2].split('_')[-1])
        except IndexError:
            self.step = 0

        logger.info('Loading checkpoint from %s' % self.args.test_from)
        checkpoint = torch.load(self.args.test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                setattr(self.args, k, opt[k])

        config = BertConfig.from_json_file(self.args.bert_config_path)
        self.model = model_builder.Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=config)
        self.model.load_cp(checkpoint)
        self.model.eval()

    def predict(self):

        test_iter = data_loader.DataLoader(self.args, data_loader.load_dataset(self.args, 'test', shuffle=False),
                                           self.args.batch_size, self.device, shuffle=False, is_test=True)
        trainer = build_trainer(self.args, self.device_id, self.model, None)
        trainer.predict(test_iter, self.step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='transformer', type=str,
                        choices=['classifier', 'transformer', 'rnn', 'baseline'])
    parser.add_argument("-data_name", default='chinese_summary', help='vy_text')
    parser.add_argument("-bert_data_path", default='./data/predict_data/', help='./data/predict_data/')
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
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./logs/project.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='./models/models_check_points/model_step_50000.pt')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-shuffle_data", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("-vy_predict", type=str2bool, nargs='?', const=False, default=True)

    _args = parser.parse_args()

    gpu_ranks: str = str(_args.gpu_ranks)
    _args.gpu_ranks = [int(i) for i in gpu_ranks.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = _args.visible_gpus

    init_logger(_args.log_file)
    _device = "cpu" if _args.visible_gpus == '-1' else "cuda"
    _device_id = 0 if _device == "cuda" else -1

    runner = Running(args=_args, device_id=_device_id)

    runner.predict()
