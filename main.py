import os
import argparse
# noinspection PyUnresolvedReferences
from managers import *
from utils import parse_config

import torch
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Set path to configuration files, e.g. '
                             'python main.py --config configs/OCRNet.json.')
    parser.add_argument('-u', '--user', type=str, default='github',
                        help='Select user to set correct data / logging paths for your system, e.g. '
                             'python main.py --user theo, to add a new user go to configs/paths_info.json and add'
                             'a new entry for ex. :'
                             '"theo":[path_to_data, path_to_log_files_and_pretrained_models]' )
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='Select GPU device to run the experiment for example --device 1')

    parser.add_argument('-dp', '--data_path', type=str, default=None,
                        help='path to data,'
                             ' if not provided this is set from the configuration file found in configs/path_info.json')

    parser.add_argument('-bl', '--blacklisting', type=bool, default=None,
                        help='remove blacklisted (mislabelled) data,'
                             ' if not provided this is set from the configuration file')

    parser.add_argument('-rl', '--use_relabeled', type=bool, default=None,
                        help='use relabelled, '
                             'if not provided this is set from the configuration file')

    parser.add_argument('-t', '--task', type=int, default=None,
                        help='sets task 1,2 or 3, '
                             'if not provided this is set from the configuration file')

    parser.add_argument('-bs', '--batch_size', type=int, default=None,
                        help='batch size for training, '
                             'if not provided this is set from the configuration file')

    args = parser.parse_args()
    config = parse_config(args.config, args.user, args.device)
    manager_class = globals()[config['manager'] + 'Manager']

    # override configuration file entries if provided with cmd line arguments
    if args.task:
        assert args.task in [1, 2, 3], f'task must be in [1,2,3] instead got {args.task}'
        config['data']['experiment'] = args.task
    if args.data_path:
        config['data_path'] = args.data_path
    if args.blacklisting:
        config['data']['blacklist'] = args.blacklisting
    if args.use_relabeled:
        config['data']['use_relabeled'] = args.use_relabeled
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    manager = manager_class(config)

    if config['mode'] == 'training':
        manager.train()
    elif config['mode'] == 'inference':
        manager.infer()
