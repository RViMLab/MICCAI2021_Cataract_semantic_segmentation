from managers.BaseManager import BaseManager
from utils import to_comb_image, t_get_confusion_matrix, t_normalise_confusion_matrix, t_get_pixel_accuracy, \
    get_matrix_fig, to_numpy, t_get_mean_iou, CLASS_INFO
import torch
from torch import nn
import numpy as np
import datetime
from models import DeepLabv3Plus
from losses import LossWrapper, LovaszSoftmax


class DeepLabv3PlusManager(BaseManager):
    """Manager for simple img in, lbl out models"""
    def load_model(self):
        """Loads the model into self.model"""
        model_class = globals()[self.config['graph']['model']]
        self.model = model_class(config=self.config['graph'], experiment=self.experiment)
        self.model = self.model.to(self.device)
        num_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Using model '{}' with backbone '{}' with output stride {} : trainable parameters {}"
              .format(self.config['graph']['model'], self.config['graph']['backbone'],  self.model.out_stride,
                      num_train_params))
        if 'graph' in self.config:
            # todo change config of upernet to have all model architecture info under 'graph' -- to avoid ifs
            if 'ss_pretrained' in self.config['graph']:
                if self.config['graph']['ss_pretrained']:
                    self.load_ss_pretrained()

    # def load_optimiser(self):
    #     """Set optimiser and if required, learning rate schedule"""
    #     self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'])
    #     if 'lr_decay_gamma' in self.config['train']:
    #         self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser,
    #                                                                 self.config['train']['lr_decay_gamma'])
    def load_loss(self):
        """Load loss function"""
        if 'loss' in self.config:
            loss_class = globals()[self.config['loss']['name']]
            self.config['loss']['experiment'] = self.experiment
            self.config['loss']['device'] = str(self.device)
            self.loss = loss_class(self.config['loss'])
            self.loss = self.loss.to(self.device)
            if isinstance(self.loss, LossWrapper):
                print("Loaded loss: " + self.loss.info_string)
            else:
                print("Loaded loss function: {}".format(loss_class))

    def train(self):
        """Main training loop"""
        print("\n***** Training started *****\n")
        for self.epoch in range(self.config['train']['epochs']):
            self.train_one_epoch()
            self.validate()
        print("\n***** Training finished *****\n"
              "Run ID: {}\n"
              "     Best validation loss: {:.5f}\n"
              "     Best mIoU         (tot / anat / ins): {:.4f} / {:.4f} / {:.4f} @ epoch {} (step {})\n"
              "     mIoU at best loss (tot / anat / ins): {:.4f} / {:.4f} / {:.4f} @ epoch {} (step {})"
              .format(self.run_id, self.best_loss, self.metrics['best_miou'], self.metrics['best_miou_anatomies'],
                      self.metrics['best_miou_instruments'], *self.metrics['best_miou_epoch_step'],
                      self.metrics['best_loss_miou'], self.metrics['best_loss_miou_anatomies'],
                      self.metrics['best_loss_miou_instruments'], *self.metrics['best_loss_epoch_step']))
        self.finalise()

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        a = datetime.datetime.now()
        running_confusion_matrix = 0
        for batch_num, (img, lbl, metadata) in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            b = (datetime.datetime.now() - a).total_seconds() * 1000
            a = datetime.datetime.now()
            img, lbl = img.to(self.device), lbl.to(self.device)
            if (self.epoch + self.start_epoch) == 0 and batch_num == 0:  # To add the graph to TensorBoard
                self.model.eval()
                self.train_writer.add_graph(self.model, img.float())
                self.model.train()
            self.optimiser.zero_grad()
            # forward
            if self.model.projector_model and isinstance(self.loss, LossWrapper):
                output, proj_features = self.model(img.float())
                loss = self.loss(proj_features, output, lbl.long())
            else:
                output = self.model(img.float())
                loss = self.loss(output, lbl.long())
            # backward
            loss.backward()
            self.optimiser.step()

            # logging
            if batch_num == 0:
                rec_num = 0  # Just take the first of the batch (will be random)
                lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                self.train_writer.add_image(
                    'train_images/record_{:02d}'.format(rec_num),
                    to_comb_image(img[rec_num], lbl[rec_num], lbl_pred[rec_num], self.config['data']['experiment']),
                    self.global_step, dataformats='HWC')

            self.train_writer.add_scalar('metrics/loss', loss.item(), self.global_step)
            info_string = ''
            if hasattr(self.loss, 'loss_vals'):
                for key in self.loss.loss_vals:
                    info_string += ' {} {:.5f}; '.format(str(key), self.loss.loss_vals[key].item())
                    self.train_writer.add_scalar('metrics/{}'.format(str(key)), self.loss.loss_vals[key].item(), self.global_step)

            confusion_matrix = t_get_confusion_matrix(output, lbl)
            running_confusion_matrix += confusion_matrix
            pa, pac = t_get_pixel_accuracy(confusion_matrix)
            iou, iou_instruments, iou_anatomies, iou_rare =\
                t_get_mean_iou(confusion_matrix, self.config['data']['experiment'],
                               categories=True, calculate_mean=False, rare=True)
            if 'train_adaptive_batching_loader' in self.train_schedule.values():
                iou_values = (1 - self.config['data']['adaptive_iou_update']) * self.metrics['iou_values'] + \
                             self.config['data']['adaptive_iou_update'] * to_numpy(iou)
                self.metrics['iou_values'][:] = iou_values

            self.train_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
            self.train_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)
            self.global_step += 1
            print("\rEpoch {:03d}, Batch {:03d} - Loss: {:.5f}; {} Time taken: {}"
                  .format(self.epoch + self.start_epoch, batch_num, loss.item(), info_string, b), end='', flush=True)
        if isinstance(self.loss, LossWrapper):
            if 'DenseContrastiveLoss' in self.loss.loss_classes:
                col_confusion_matrix = t_normalise_confusion_matrix(running_confusion_matrix, mode='col')
                self.train_writer.add_figure('train_confusion_matrix/col_normalised',
                                             get_matrix_fig(to_numpy(col_confusion_matrix),
                                                            self.config['data']['experiment']),
                                             self.global_step - 1)
                self.loss.loss_classes['DenseContrastiveLoss'].update_confusion_matrix(col_confusion_matrix)
        if self.scheduler is not None:
            self.scheduler.step()
            self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step)

    def validate(self):
        """Validate the model on the validation data"""
        self.model.eval()
        valid_loss = 0
        confusion_matrix = None
        individual_losses = dict()
        if self.model.projector_model and isinstance(self.loss, LossWrapper):
            for key in self.loss.loss_vals:
                individual_losses[key] = 0

        with torch.no_grad():
            for rec_num, (img, lbl, metadata) in enumerate(self.data_loaders['valid_loader']):
                img, lbl = img.to(self.device), lbl.to(self.device)

                if self.model.projector_model and isinstance(self.loss, LossWrapper):
                    output, proj_features = self.model(img.float())
                    valid_loss += self.loss(proj_features, output, lbl.long()).item()
                    for key in self.loss.loss_vals:
                        individual_losses[key] += self.loss.loss_vals[key]
                else:
                    output = self.model(img.float())
                    valid_loss += self.loss(output, lbl.long()).item()
                confusion_matrix = t_get_confusion_matrix(output, lbl, confusion_matrix)

                if rec_num in np.round(np.linspace(0, len(self.data_loaders['valid_loader']) - 1, self.max_valid_imgs)):
                    lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                    self.valid_writer.add_image(
                        'valid_images/record_{:02d}'.format(rec_num),
                        to_comb_image(img[0], lbl[0], lbl_pred[0], self.config['data']['experiment']),
                        self.global_step, dataformats='HWC')

        valid_loss /= len(self.data_loaders['valid_loader'])

        # logging
        self.valid_writer.add_scalar('metrics/loss', valid_loss, self.global_step - 1)
        info_string = ''
        if hasattr(self.loss, 'loss_vals'):
            for key in self.loss.loss_vals:
                individual_losses[key] /= len(self.data_loaders['valid_loader'])
                info_string += ' {} {:.5f}; '.format(str(key), individual_losses[key].item())
                self.valid_writer.add_scalar('metrics/{}'.format(str(key)), individual_losses[key].item(),
                                             self.global_step - 1)

        row_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'row')
        col_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'col')
        self.valid_writer.add_figure('valid_confusion_matrix/row_normalised',
                                     get_matrix_fig(to_numpy(row_confusion_matrix), self.config['data']['experiment']),
                                     self.global_step - 1)
        self.valid_writer.add_figure('valid_confusion_matrix/col_normalised',
                                     get_matrix_fig(to_numpy(col_confusion_matrix), self.config['data']['experiment']),
                                     self.global_step - 1)
        pa, pac = t_get_pixel_accuracy(confusion_matrix)
        m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare = t_get_mean_iou(confusion_matrix,
                                                                               self.config['data']['experiment'], True,
                                                                               rare=True)

        self.valid_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
        self.valid_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou', m_iou, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_anatomies', m_iou_anatomies, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_instruments', m_iou_instruments, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou_rare', m_iou_rare, self.global_step)

        print("\rEpoch {:03d} - Validation loss: {:.5f} - {} - miou:{:.3f} - ins:{:.3f} -"
              " anat:{:.3f} - rare:{:.4f}".format(self.epoch + self.start_epoch, valid_loss, info_string,
                                                  m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare))
        m_iou = round(float(m_iou.cpu().numpy()), 4)
        m_iou_instruments = round(float(m_iou_instruments.cpu().numpy()), 4)
        m_iou_anatomies = round(float(m_iou_anatomies.cpu().numpy()), 4)

        best_miou_flag = False
        if m_iou > self.metrics['best_miou']:
            self.metrics.update({'best_miou': m_iou,
                                 'best_miou_anatomies': m_iou_anatomies,
                                 'best_miou_instruments': m_iou_instruments,
                                 'best_miou_rare': m_iou_rare,
                                 'best_miou_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})

            best_miou_flag = True
            print("            New best mIoU (tot / anat / ins): {:.4f} / {:.4f} / {:.4f}"
                  .format(m_iou, m_iou_anatomies, m_iou_instruments))
            self.save_checkpoint(is_best=True)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.metrics.update({'best_loss_miou': m_iou,
                                 'best_loss_miou_anatomies': m_iou_anatomies,
                                 'best_loss_miou_instruments': m_iou_instruments,
                                 'best_loss_miou_rare': m_iou_rare,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})

            print("            New best validation loss: {:.5f}".format(valid_loss))
            if not best_miou_flag:
                print("            --- with mIoU (tot / anat / ins): {:.4f} / {:.4f} / {:.4f}\n"
                      "            --- best mIoU (tot / anat / ins): {:.4f} / {:.4f} / {:.4f}"
                      .format(m_iou, m_iou_anatomies, m_iou_instruments, self.metrics['best_miou'],
                              self.metrics['best_miou_anatomies'], self.metrics['best_miou_instruments']))

        if self.epoch % self.config['log_every_n_epochs'] == 0 and self.epoch > 0\
                or self.epoch == self.config['train']['epochs'] - 1:
            self.save_checkpoint(is_best=False)

        # Update info.json file so it exists in case the run stops / crashes before self.finalise()
        self.write_info_json()
