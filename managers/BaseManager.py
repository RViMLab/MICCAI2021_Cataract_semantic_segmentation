import pathlib
import cv2
import json
import torch
import datetime
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as f
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
from datasets import DatasetFromDF, DatasetFromVideo
# noinspection PyUnresolvedReferences
from models import *
# from backboned_unet import Unet  # conda env update -f env_server_2.yml
# noinspection PyUnresolvedReferences
from losses import *
from utils import DATA_SPLITS, CLASS_INFO, OVERSAMPLING_PRESETS, parse_transform_list, get_class_info, \
    AdaptiveBatchSampler, RepeatFactorSampler, LRFcts, mask_to_colormap, get_remapped_colormap, \
    t_get_mean_iou, t_get_confusion_matrix, to_comb_image, to_numpy, worker_init_fn


class BaseManager:
    """Base Manager class, from which all other managers inherit"""
    def __init__(self, configuration):
        """Sets up the run manager, either for training or for inference

        :param configuration: dict with the following keys:
            mode: 'training' or 'inference' or 'video_inference'
            graph: dict with key 'model' (optionally 'width' etc.), 'loss' (optionally 'multi_scale_loss' etc.)
            data: dict with keys 'named_dataset'/'dataset_list', 'train_split', 'batch_size'
            train: dict with keys 'learning_rate', 'epochs'
            data_path: Base path to where the data is found (in original CaDISv2 format with folders etc.)
            log_path: Base path to where checkpoints etc are logged
            log_every_n_epochs: How often a checkpoint is logged
            cuda: 'true'/'false' - whether the model runs on GPU or CPU
            gpu_device: if cuda==True, then which device is to be used
            seed: torch random seed
            infer_config keys: mode: 'inference', model, width, log_path, log_name, checkpoint_type,
                               cuda, gpu_device (if cuda)
        """
        # Set up parameters
        self.config = configuration
        self.debugging = self.config['debugging']
        self.start_epoch = 0
        self.epoch = 0
        self.best_loss = 1e10
        self.metrics = {'best_miou': 0,
                        'best_miou_anatomies': 0,
                        'best_miou_instruments': 0,
                        'best_miou_rare': 0,
                        'best_miou_epoch_step': 'n/a'}
        self.global_step = 0
        self.max_valid_imgs = self.config['max_valid_imgs']  # Maximum number of images saved to tensorboard
        self.experiment = self.config['data']['experiment']
        self.data_preloaded = self.config['data']['preload']
        self.num_classes = len(CLASS_INFO[self.experiment][0])
        self.model = None
        self.tta_model = None
        self.data_loaders = {}
        self.loss = 0
        self.optimiser = None
        self.scheduler = None
        self.train_schedule = {}
        self.save_dir_path = None  # path to where pseudo labelled data are saved
        for i in range(self.config['train']['epochs']):
            self.train_schedule.update({i: 'train_loader'})  # pre-fill

        # Print debugging state in Console
        if self.debugging:
            print("\n\n* * * * * DEBUGGING ACTIVE * * * * * \n\n")

        # Set cuda flag
        if torch.cuda.is_available() and not self.config['cuda']:
            print("WARNING: CUDA device available, but not used")
        if self.config['cuda'] and not torch.cuda.is_available():
            print("WARNING: CUDA device required, but not available - using CPU instead")
        self.cuda = torch.cuda.is_available() & self.config['cuda']

        if self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config['gpu_device'])
            print("Program will run on *****GPU-CUDA, device {}*****".format(self.config['gpu_device']))
        else:
            self.device = torch.device('cpu')
            print("Program will run on *****CPU*****")

        # Identify run
        if 'load_checkpoint' in self.config and self.config['mode'] is not 'training':
            self.run_id = self.config['load_checkpoint']
        else:
            self.run_id = '{:%Y%m%d_%H%M%S}_e{}'.format(datetime.datetime.now(), self.experiment)
            if 'name' in self.config:
                self.run_id = '__'.join((self.run_id, self.config['name']))
        self.log_dir = pathlib.Path(self.config['log_path']) / self.run_id
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)
        print("Run ID: {}".format(self.run_id))

        # Load model into self.model
        self.load_model()

        if self.config['mode'] == 'training':

            # Load loss into self.loss
            self.load_loss()

            # Set manual seeds to make training repeatable
            torch.manual_seed(self.config['seed'])

            # Load the datasets if given
            self.load_data()

            # Optimiser
            self.load_optimiser()

            # Tensorboard writers
            self.train_writer = SummaryWriter(log_dir=self.log_dir / 'train')
            self.valid_writer = SummaryWriter(log_dir=self.log_dir / 'valid')

        elif self.config['mode'] == 'video_inference':
            # this mode loads mp4 dataset and performs inference
            torch.manual_seed(self.config['seed'])
            self.video_info = dict()

        elif self.config['mode'] == 'demo_video_inference':
            # must set mode to demo_video_inference
            # must have video_ids as key in config and a list of [devxxx, ...] 
            torch.manual_seed(self.config['seed'])
            self.video_info = dict()
            self.load_data()
            self.demo_infer()

        elif self.config['mode'] == 'inference':
            # this mode loads only the validation dataset of a split
            self.valid_writer = SummaryWriter(log_dir=self.log_dir / 'infer')
            torch.manual_seed(self.config['seed'])
            self.load_data()

        else:
            raise ValueError(f'mode: {self.config["mode"]} is not recognized')

    def load_data(self):
        """Creates a dict containing the training and validation data loaders, loaded into self.data"""
        if self.config['mode'] == 'demo_video_inference':
            print('***** going to run on workflow/test/dev0*.mp4 videos and save outputs as video*****')
            assert 'video_ids' in self.config, 'missing key video_ids in config ' \
                                               'for example config[video_ids] = [dev01, dev02]'
            if 'demo_frame_freq' not in self.config:
                self.config['demo_frame_freq'] = 1  # i.e
            print('***** going to run model on every {}-th frame  *****'.format(self.config['demo_frame_freq']))
            print('                       of videos: {}           *****'.format(self.config['video_ids']))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # todo enable path_mp4_dir to be externally set with the below being the default
            path_mp4_dir = pathlib.Path(self.config['data_path']).parent / \
                pathlib.Path('workflow') / pathlib.Path('test')
            for path_to_mp4_file in path_mp4_dir.glob('**/*.mp4'):
                name = path_to_mp4_file.name
                if 'miccai_demo' in self.config:
                    # in this mode it generates prediction videos
                    video_writer_shape = (960, 540)
                else:
                    # else it concats input and prediction horrizontaly 
                    video_writer_shape = (2*960, 540)

                if name.split('.')[0] in self.config['video_ids']:
                    print('preparing capture and writer for {}'.format(path_to_mp4_file))
                    capture = cv2.VideoCapture(str(path_to_mp4_file))
                    frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_ids = np.arange(0, frame_cnt, dtype=np.int32).tolist()
                    frame_ids = frame_ids[0::self.config['demo_frame_freq']]
                    video_id = int(name.split('.')[0].split('dev')[-1])  # get the XX out of devXX.mp4
                    self.video_info[video_id] = dict()  # frame count
                    self.video_info[video_id]['frame_count'] = frame_cnt
                    self.video_info[video_id]['frame_ids'] = frame_ids  # frames that will be read
                    self.video_info[video_id]['capture'] = capture
                    self.video_info[video_id]['writer'] =\
                        cv2.VideoWriter(str(self.log_dir / '{}_{}.avi'.format(name.split('.')[0],
                                                                              self.config['graph']['model'])),
                                        fourcc, 30, video_writer_shape)

            video_dataset = DatasetFromVideo(video_info=self.video_info, frame_h=540, frame_w=960, get_id=True)
            video_frame_loader = DataLoader(video_dataset, 1, False, worker_init_fn=worker_init_fn)
            self.data_loaders.update({'video_inference_loader': video_frame_loader})
            return

        # Create default dataloaders
        if self.config['mode'] == 'inference':
            train_df, valid_df = self.get_seg_dataframes()
            _, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')
            self.data_loaders = {'valid_loader': valid_loader}
            return

        train_df, valid_df = self.get_seg_dataframes()
        train_loader, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')
        self.data_loaders = {'train_loader': train_loader,
                             'valid_loader': valid_loader}
        # Obtain schedule
        loader_type_list = ['adaptive_batching', 'oversampling', 'weighted_random', 'repeat_factor']
        l_list = [self.config['data'][loader_type][0] for loader_type in loader_type_list]
        idx = np.argsort(l_list)
        for loader_type in np.array(loader_type_list)[idx]:
            loader = 'train_' + loader_type + '_loader'
            if len(self.config['data'][loader_type]) == 1:
                self.config['data'][loader_type].extend([self.config['train']['epochs']])
            for i in range(*self.config['data'][loader_type]):
                self.train_schedule[i] = loader
            if loader in self.train_schedule.values():
                self.data_loaders.update({loader: self.get_dataloaders(train_df, valid_df, loader)})
        # Print schedule
        print("Training schedule created:")
        start, stop = None, None
        for i in range(1, self.config['train']['epochs']):
            if start is None:
                start = i - 1
            if self.train_schedule[i] != self.train_schedule[i - 1]:
                stop = i - 1
            elif i == self.config['train']['epochs'] - 1:
                stop = i
            if start is not None and stop is not None:
                if start == stop:
                    print("          Epoch {}: {}".format(start, self.train_schedule[i - 1]))
                else:
                    print("    Epochs {} to {}: {}".format(start, stop, self.train_schedule[i - 1]))
                start, stop = None, None

    def get_seg_dataframes(self):
        """Creates the training and validation segmentation datasets from the config"""
        # Make dataframes for the training and the validation set
        df = pd.read_csv('data/data.csv')
        if 'random_split' in self.config['data']:
            print("***Legacy mode: random split of all data used, instead of split of videos!***")
            train = df.sample(frac=self.config['data']['random_split'][0]).copy()
            valid = df.drop(train.index).copy()
            split_of_rest = self.config['data']['random_split'][1] / (1 - self.config['data']['random_split'][0])
            valid = valid.sample(frac=split_of_rest)
        else:
            split = DATA_SPLITS[int(self.config['data']['split'])]
            if len(split) == 2:
                train_videos, valid_videos = split
            else:
                train_videos, valid_videos, test_videos = split
                if self.config['mode'] == 'inference':
                    valid_videos = test_videos # replace valid_videos with test_videos for infer mode
                    print('using a train-val-test split with mode = {} --> infer to be run on test set'.format(self.config['mode']))
            train = df.loc[df['vid_num'].isin(train_videos)].copy()
            valid = df.loc[df['vid_num'].isin(valid_videos)].copy()  # No prop lbl in valid
        info_string = "Dataframes created. Number of records training / validation: {:06d} / {:06d}\n"\
                      "                    Actual data split training / validation: {:.3f}  / {:.3f}"\
            .format(len(train.index), len(valid.index), len(train.index) / len(df), len(valid.index) / len(df))

        # Replace incorrectly annotated frames if flag set
        if self.config['data']['use_relabeled']:
            train_idx_list = train[train['relabeled'] == 1].index
            for idx in train_idx_list:
                train.loc[idx, 'blacklisted'] = 0  # So the frames don't get removed after
                lbl_path = pathlib.Path(train.loc[idx, 'lbl_path']).name
                train.loc[idx, 'lbl_path'] = 'relabeled/' + str(lbl_path)
            valid_idx_list = valid[valid['relabeled'] == 1].index
            for idx in valid_idx_list:
                valid.loc[idx, 'blacklisted'] = 0  # So the frames don't get removed after
                lbl_path = pathlib.Path(valid.loc[idx, 'lbl_path']).name
                valid.loc[idx, 'lbl_path'] = 'relabeled/' + str(lbl_path)
            info_string += "\n                                       Relabeled train recs: {}\n"\
                           "                                       Relabeled valid recs: {}"\
                .format(len(train_idx_list), len(valid_idx_list))

        # Remove incorrectly annotated frames if flag set
        if self.config['data']['blacklist']:
            train = train.drop(train[train['blacklisted'] == 1].index)
            valid = valid.drop(valid[valid['blacklisted'] == 1].index)
            t_len, v_len = len(train.index), len(valid.index)
            info_string += "\n        After blacklisting: Number of records train / valid: {:06d} / {:06d}\n"\
                           "                          Relative data split train / valid: {:.3f}  / {:.3f}"\
                .format(t_len, v_len, t_len / (t_len + v_len), v_len / (t_len + v_len))
        train = train.reset_index()
        valid = valid.reset_index()
        # Console output
        print(info_string)
        return train, valid

    def get_dataloaders(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,  mode: str = 'default', **kwargs):
        transforms_dict = parse_transform_list(self.config['data']['transforms'],
                                               self.config['data']['transform_values'],
                                               self.num_classes)
        # Dataset transforms console output
        img_transforms = [str(type(item).__name__) for item in transforms_dict['train']['img'] if
                          not (isinstance(item, ToPILImage) or isinstance(item, ToTensor))]
        common_transforms = [str(type(item).__name__) for item in transforms_dict['train']['common']]
        print("Dataset transforms: {}".format(img_transforms + common_transforms))

        data_path = None if self.data_preloaded else self.config['data_path']
        real_num_classes = self.num_classes if self.experiment == 1 else self.num_classes - 1
        num_workers = int(self.config['data']['num_workers'])
        if mode == 'default':
            train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                      data_path=data_path)
            valid_set = DatasetFromDF(valid_df, self.experiment, transforms_dict['valid'], data_path=data_path)
            train_loader = DataLoader(train_set, batch_size=self.config['data']['batch_size'],
                                      shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn)
            valid_loader = DataLoader(valid_set, num_workers=num_workers, worker_init_fn=worker_init_fn)
            print("Dataloaders created. Batch size: {}\n"
                  "              Number of workers: {}\n"
                  .format(self.config['data']['batch_size'], num_workers))
            return train_loader, valid_loader
        elif mode == 'train_adaptive_batching_loader':
            train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                      data_path=data_path)
            self.metrics['iou_values'] = np.ones(real_num_classes, 'f') * .5
            df = get_class_info(train_df, self.experiment)
            s = self.config['data']['adaptive_sel_size']
            sampler = AdaptiveBatchSampler(data_source=train_set, dataframe=df, iou_values=self.metrics['iou_values'],
                                           dist_type='1-**2', batch_size=self.config['data']['batch_size'], sel_size=s)
            train_adaptive_batching_loader = DataLoader(train_set,
                                                        batch_sampler=sampler,
                                                        num_workers=num_workers,
                                                        worker_init_fn=worker_init_fn)
            print("Adaptive batching dataloader created. Selection size: {}\n"
                  "                                IoU update per batch: {}\n"
                  .format(s, self.config['data']['adaptive_iou_update']))
            return train_adaptive_batching_loader
        elif mode == 'train_oversampling_loader':
            preset = self.config['data']['oversampling_preset']
            class_list = OVERSAMPLING_PRESETS[preset][self.experiment - 1]
            df = get_class_info(train_df, self.experiment)
            required_num = int(len(train_df.index) * self.config['data']['oversampling_frac'])
            train_extend = []
            sel_per_class_estimate = required_num // len(class_list)
            while len(train_extend) < required_num:
                series_list = []
                for c in class_list:
                    series_list.append(df.sort_values(by=c, axis=0, ascending=False).iloc[:sel_per_class_estimate])
                train_extend = pd.concat(series_list).drop_duplicates()
                sel_per_class_estimate += np.maximum(1, (required_num - len(train_extend)) // len(class_list))
            train_df = pd.concat([train_df, train_extend], join='inner').reset_index()
            train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                      data_path=data_path)
            train_oversampling_loader = DataLoader(train_set, batch_size=self.config['data']['batch_size'],
                                                   shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn)
            print("Oversampling dataloader created. Oversampling fraction: {:.2f}\n"
                  "                                   Oversampling preset: {}\n"
                  "                                  Oversampling classes: {}\n"
                  "                      Resulting total training records: {:06d}"
                  .format(self.config['data']['oversampling_frac'], preset, class_list, len(train_df.index)))
            return train_oversampling_loader
        elif mode == 'train_weighted_random_loader':
            train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                      data_path=data_path)
            # Get class absolute and relative incidence, per record and overall
            df = get_class_info(train_df, self.experiment)
            class_abs = df[list(range(real_num_classes))].values.astype('f')
            # class_rel_to_record = class_abs / np.sum(class_abs, axis=1)[:, np.newaxis]
            class_rel_to_class_sum = class_abs / np.sum(class_abs, axis=0)
            class_freq = np.sum(class_abs, axis=0) / np.sum(class_abs)
            # Calculate a weighting for each class
            mode = self.config['data']['weighted_random_mode']
            weights = None
            if mode == 'v1':
                class_weighting = 1 / class_freq
                class_weighting /= np.sum(class_weighting)
                weights = np.sum(class_abs * class_weighting[np.newaxis, :], axis=1)
            elif mode == 'v2':
                # "The more there is of a class w.r.t. the totality of the class in the dataset, and
                # the rarer this class is in the totality of the dataset"
                class_weighting = 1 - class_freq
                weights = np.sum(class_rel_to_class_sum * class_weighting, axis=1)
            else:
                ValueError("Mode '{}' not recognised.".format(mode))
            sampler = WeightedRandomSampler(weights, len(train_set))
            train_weighted_random_loader = DataLoader(train_set, batch_size=self.config['data']['batch_size'],
                                                      sampler=sampler, num_workers=num_workers,
                                                      worker_init_fn=worker_init_fn)
            print("Weighted random dataloader created.\n")
            return train_weighted_random_loader
        elif mode == 'train_repeat_factor_loader':

            train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                      data_path=data_path)

            sampler = RepeatFactorSampler(data_source=train_set, dataframe=train_df,
                                          repeat_thresh=self.config['data']['repeat_factor_freq_thresh'],
                                          experiment=self.config['data']['experiment'],
                                          split=int(self.config['data']['split']))
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=self.config['data']['batch_size'],
                                                                  drop_last=True)

            train_repeat_factor_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=0,
                                                    worker_init_fn=worker_init_fn)
            # valid_set = DatasetFromDF(valid_df, self.experiment, transforms_dict['valid'], data_path=data_path)
            # valid_loader = DataLoader(valid_set, num_workers=num_workers, worker_init_fn=worker_init_fn)
            img_rfs = sampler.repeat_factors.numpy()
            # cls_rfs = sampler.class_repeat_factors
            # cls_rfs = {k: v for k, v in sorted(cls_rfs.items(), reverse=True, key=lambda item: item[1])}
            frames_repeated = sum(img_rfs[img_rfs > 1])
            print("Repeat factor dataloader created. frequency threshold: {:.2f}\n"
                  "                                  frames with rf>1:  {}\n"
                  "                      Resulting total training records (aprox): {:.2f}"
                  .format(self.config['data']['repeat_factor_freq_thresh'], frames_repeated, sum(img_rfs)))
            return train_repeat_factor_loader
        else:
            ValueError("Dataloader special type '{}' not recognised".format(mode))

    def load_model(self):
        """Loads the model into self.model"""
        model_class = globals()[self.config['graph']['model']]
        self.model = model_class(config=self.config['graph'], experiment=self.experiment)
        self.model = self.model.to(self.device)
        num_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Using model '{}', width {:.3f}: {} trainable parameters"
              .format(self.config['graph']['model'], self.config['graph']['width'], num_train_params))

        if 'graph' in self.config:
            # todo change config of upernet to have all model architecture info under 'graph' -- to avoid ifs
            if 'ss_pretrained' in self.config['graph']:
                if self.config['graph']['ss_pretrained']:
                    self.load_ss_pretrained()

    def load_loss(self):
        """Load loss function"""
        if 'loss' in self.config:
            loss_class = globals()[self.config['loss']['name']]
            self.config['loss']['experiment'] = self.experiment
            self.config['loss']['device'] = str(self.device)
            self.loss = loss_class(self.config['loss'])
            self.loss = self.loss.to(self.device)
            print("Loaded loss function: {}".format(loss_class))
        else:
            # by default ignore_class is ignored by the loss for experiments 2 and 3
            # -100 is the default of nn.CrossEntropyLoss for ignore_index
            ignore_index_in_loss = len(CLASS_INFO[self.config['data']['experiment']][1]) - 1 if \
                self.experiment in [2, 3] else -100
            self.loss = CrossEntropyLoss(ignore_index=ignore_index_in_loss)
            print("Loaded loss function: Cross Entropy")

    def load_optimiser(self):
        """Set optimiser and if required, learning rate schedule"""
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'])
        if self.config['train']['lr_batchwise']:  # Replace given lr_restarts with numbers in batches instead of epochs
            b_per_e = [len(self.data_loaders[self.train_schedule[e]]) for e in range(self.config['train']['epochs'])]
            lr_total_steps = np.sum(b_per_e)
            r = self.config['train']['lr_restarts']
            new_r = []
            if len(r) > 0:
                r.insert(0, 0)
                for i in range(len(r) - 1):
                    new_r.append(int(np.sum(np.array(b_per_e)[r[i]:r[i + 1]]) + np.sum(new_r[:i])))
            lr_restart_steps = new_r
            # # Adjust params for exponential decay. This is experimental - adjustment is such that the decay over steps
            # # in the very first epoch will equal a decay of lr_params in that first epoch. If later epochs use a
            # # different number of steps, this is not taken into account.
            # self.config['train']['lr_params'] = np.power(self.config['train']['lr_params'], 1 / b_per_e[0])
        else:
            lr_restart_steps = self.config['train']['lr_restarts']
            lr_total_steps = self.config['train']['epochs']
            print("*** lr_schedule: '{}' over total epochs {} with restarts @ {}"
                  .format(self.config['train']['lr_fct'], lr_total_steps, lr_restart_steps))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=LRFcts(self.config['train'],
                                                                                            lr_restart_steps,
                                                                                            lr_total_steps))
        # lrs = [(self.scheduler.step(), self.scheduler.get_lr()[0])[1] for i in range(self.config['train']['epochs'])]
        # import matplotlib.pyplot as plt
        # plt.plot(lrs)
        # plt.show()
        # a=1

    def save_checkpoint(self, is_best):
        """Saves a checkpoint in given self.log_dir

        :param is_best: Determines whether the checkpoint is a current best
        """
        base_path = self.log_dir / 'chkpts'
        if not base_path.is_dir():
            base_path.mkdir()
        state = {
            'global_step': self.global_step,
            'epoch': self.start_epoch + self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'best_loss': self.best_loss,
            'best_miou': self.metrics['best_miou'],
            'is_best': is_best
        }
        if self.scheduler is not None:
            state.update({'scheduler_state_dict': self.scheduler.state_dict()})
        if is_best:
            name = 'chkpt_best.pt'
        else:
            name = 'chkpt_epoch_{:03d}.pt'.format(state['epoch'])
        torch.save(state, base_path / name)
        print("Checkpoint saved: {}".format(name))

    def load_checkpoint(self, chkpt_type):
        """Load a model and model state from a checkpoint

        :param chkpt_type: 'best' or 'last'
        :return:
        """
        checkpoint_list = [f.name for f in (self.log_dir / 'chkpts').iterdir()]
        checkpoint_list.sort()
        name = 'chkpt_best.pt'
        if chkpt_type == 'best':
            if name not in checkpoint_list:
                raise ValueError("No checkpoint of type 'best' found.")
        elif chkpt_type == 'last':
            if 'chkpt_epoch_' in checkpoint_list[-1]:
                name = checkpoint_list[-1]
            else:
                raise ValueError("No checkpoint of type 'last' found.")
        path = self.log_dir / 'chkpts' / name
        # print(torch.cuda.current_device())
        # this is required if it checkpoint trained on one device and now is loaded on a different device
        # https://github.com/pytorch/pytorch/issues/15541
        map_location = 'cuda:{}'.format(self.config['gpu_device'])
        checkpoint = torch.load(str(path), map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False) # todo fix this
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict']) if self.config['mode'] == 'training' \
            else None
        if 'scheduler_state_dict' in checkpoint and self.config['mode'] == 'training':
            self.scheduler.load_state_dict(checkpoint['optimiser_state_dict'])
        self.start_epoch = checkpoint['epoch'] if self.config['mode'] == 'training' else None
        self.global_step = checkpoint['global_step'] if self.config['mode'] == 'training' else None
        self.best_loss = checkpoint['best_loss']
        self.metrics['best_miou'] = checkpoint['best_miou']
        print("Checkpoint loaded: {}".format(path))


    def load_ss_pretrained(self):
        """ method that initializes a resnet backbone with self-supervised pretrained weights
            to use add in config: "graph": {"pretrained": false, # does not load imagenent checkpoint from torchvision
                                            "ss_pretrained": "moco"}
            assumes that self.model stores the network and it has a self.model.backbone for the resnet encoder

           --for now only resnet 50 is supported !!
           --for now only moco resnet 50 is supported, weights from official repo: moco_v2_800ep_pretrain.pth.tar

        """
        assert (hasattr(self, 'model'))
        assert (hasattr(self.model, 'backbone'))
        assert self.config['graph']['ss_pretrained'] in ['moco', 'simclr'],\
            'invalid ss_pretrained {} please request any of {}'.format(self.config['graph']['ss_pretrained'],
                                                                       ['moco', 'simclr'])
        if self.config['graph']['ss_pretrained'] == 'moco':

            chkpt_path = pathlib.Path(self.config['ss_pretrained_path']) / pathlib.Path('moco')\
                         / pathlib.Path('moco_v2_800ep_pretrain.pth.tar')
            if chkpt_path.exists():
                print("=> loading checkpoint '{}'".format(chkpt_path))
                checkpoint = torch.load(chkpt_path, map_location="cpu")
                # deeplab_state_dict = self.model.state_dict()
                # backbone_state_dict = self.model.backbone.state_dict()
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                # init bacbone with ss checkpoint
                msg = self.model.backbone.load_state_dict(state_dict, strict=False)
                # after_loading_state_dict = self.model.state_dict()
                assert len(msg.missing_keys) == 0, 'key mismatch in state_dict: {}'.format(msg.missing_keys)
                print("=> loaded pre-trained model '{}'".format(chkpt_path))
            else:
                print("=> no checkpoint found at '{}'".format(chkpt_path))

    def train(self):
        """Main training loop"""
        print("\n***** Training started *****\n")
        for self.epoch in range(self.config['train']['epochs']):
            self.train_one_epoch()
            self.validate()
        print("\n***** Training finished *****\n"
              "     Best validation loss: {:.5f}\n"
              "     Best mean IoU:        {:.5f}"
              .format(self.best_loss, self.metrics['best_miou']))
        self.finalise()

    def train_one_epoch(self):
        """Train the model for one epoch"""
        raise NotImplementedError

    def validate(self):
        """Validate the model on the validation data"""
        raise NotImplementedError

    def finalise(self):
        """Saves info, resets main variables"""
        config_text = self.write_info_json()
        # Save extra info to tensorboard
        self.train_writer.add_text('info', config_text.replace('\n', '  \n'), self.global_step)
        self.train_writer.close()
        self.valid_writer.close()
        # Reset main variables
        self.run_id = None
        self.start_epoch = 0
        self.epoch = 0
        self.best_loss = 1e10
        self.metrics = {'best_miou': 0}
        self.global_step = 0

    def write_info_json(self):
        config = self.config.copy()
        config['run_id'] = self.run_id
        config['best_loss'] = self.best_loss
        metrics = self.metrics.copy()
        for k in metrics.keys():
            if isinstance(self.metrics[k], np.ndarray) or isinstance(metrics[k], torch.Tensor):
                # noinspection PyUnresolvedReferences
                metrics[k] = metrics[k].tolist()
        config['metrics'] = metrics
        # Save config to json
        config_text = json.dumps(config, indent=4, sort_keys=True, default=self.default)
        with open(self.log_dir / 'info.json', 'w') as json_file:
            json_file.write(config_text)
        return config_text

    @staticmethod
    def default(obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError('Not serializable')

    def _preload_data(self, df):
        image_list, label_list = [], []
        p = self.config['data_path']
        for i in range(len(df)):
            image_list.append(cv2.imread(str(pathlib.Path(p) / df.iloc[i].loc['img_path']))[..., ::-1])
            label_list.append(cv2.imread(str(pathlib.Path(p) / df.iloc[i].loc['lbl_path']), 0))
        df['image'] = image_list
        df['label'] = label_list
        return df

    def infer(self):
        """run the model on validation data of a split , creates a logfile named 'infer' in logging dir """
        self.model.eval()
        if hasattr(self.model, 'get_intermediate'):
            self.model.get_intermediate = False  # to supress ocr output

        if hasattr(self.model, 'get_features'):
            self.model.get_features = False  # to supress upn features

        if not isinstance(self.model, Ensemble):
            self.load_checkpoint('best')

        if self.config['tta']:
            import ttach
            transforms = ttach.Compose(
                [
                    ttach.HorizontalFlip(),
                    ttach.Scale(scales=[0.75, 1, 1.5, 1.75, 2]),
                ]
            )
            tta_model = ttach.SegmentationTTAWrapper(self.model, transforms, merge_mode='mean')

        confusion_matrix = None
        with torch.no_grad():
            for rec_num, (img, lbl, metadata) in enumerate(self.data_loaders['valid_loader']):
                print("\r Inference on {}".format(rec_num), end='', flush=True)
                img, lbl = img.to(self.device), lbl.to(self.device)
                # noinspection PyUnboundLocalVariable
                output = self.model(img.float()) if not self.config['tta'] else tta_model(img.float())
                confusion_matrix = t_get_confusion_matrix(output, lbl, confusion_matrix)
                if rec_num in np.round(np.linspace(0, len(self.data_loaders['valid_loader']) - 1, self.max_valid_imgs)):
                    if not isinstance(self.model, Ensemble):
                        lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                    else:
                        lbl_pred = torch.argmax(output, dim=1)  # already softmaxed and merged in ensemble.forward()
                    self.valid_writer.add_image(
                        'valid_images/record_{:02d}'.format(rec_num),
                        to_comb_image(img[0], lbl[0], lbl_pred[0], self.config['data']['experiment']),
                        self.global_step, dataformats='HWC')

        m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare = t_get_mean_iou(confusion_matrix,
                                                                               self.config['data']['experiment'], True,
                                                                               rare=True)
        self.valid_writer.add_scalar('metrics/mean_iou', m_iou, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_anatomies', m_iou_anatomies, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_instruments', m_iou_instruments, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_rare', m_iou_rare, self.global_step)
        print("\n miou:{:.4f} - miou-instruments{:.4f} - miou-anatomies{:.4f} - miou-rare{:.4f}".format(m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare))
        self.valid_writer.close()

    def demo_infer(self):
        """ to be used when  config[mode] = demo_video_inference
            run the model on videos and save outputs and input as a merged video
        """
        def to_comb_image_demo(img, lbl, debug=False):
            """ receives numpy arrays of shapes: img (H, W, 3), lbl (already in colormap) (H,W,3)
                returns combined numpy array img of shape (H, W*2, 3) -- combo img for demo
            """
            lbl = lbl.astype('uint8')
            # img = img.astype('uint8')
            img = np.round(np.moveaxis(img, 0, -1) * 255).astype('uint8')[..., ::-1]
            comb_img = np.concatenate((img, lbl), axis=1)
            if debug:
                cv2.imshow('comb_img_lbl', comb_img)
                cv2.waitKey(delay=2)
            return comb_img

        if hasattr(self.model, 'get_intermediate'):
            self.model.get_intermediate = False  # to supress ocr output
        if not isinstance(self.model, Ensemble):
            self.load_checkpoint('best')
        self.model.eval()
        with torch.no_grad():
            for rec_num, (frame, frame_idx, vid_id) in enumerate(self.data_loaders['video_inference_loader']):
                print("Running on video '{}' frame '{}' (#{})".format(vid_id, frame_idx, rec_num))
                frame = frame.to(self.device)
                output = self.model(frame.float())
                # upernet exception
                if isinstance(output, tuple) and (self.config['graph']['model'] == 'UPerNet'):
                    output = output[-1]

                if not isinstance(self.model, Ensemble):
                    pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                else:
                    pred = torch.argmax(output, dim=1)  # already softmaxed and merged in ensemble.forward()

                pred = to_numpy(pred)[0]
                frame = to_numpy(frame)[0]
                vid_id = to_numpy(vid_id)[0]

                # if 'pad' in self.config['data']['transforms']:  # If input img padded, output needs to be 'de-padded'
                #     pred = pred[2:-2]
                #     frame= frame[:, 2:-2, :]
                # cv2.imwrite(str(pathlib.Path(self.config['data_path']) / 'output' / img_name[0]) + '.png', pred)
                debug_pred = mask_to_colormap(pred, get_remapped_colormap(CLASS_INFO[self.experiment][0]),
                                              from_network=True, experiment=self.experiment)[..., ::-1]
                # cv2.imwrite(str(pathlib.Path(self.config['data_path']) / 'debug' / img_name[0]) + '.png', debug_pred)
                if 'miccai_demo' in self.config:
                    self.video_info[vid_id]['writer'].write(debug_pred, debug=False)
                else:
                    self.video_info[vid_id]['writer'].write(to_comb_image_demo(frame, debug_pred, debug=False))
        print("demo video inference done!.")
