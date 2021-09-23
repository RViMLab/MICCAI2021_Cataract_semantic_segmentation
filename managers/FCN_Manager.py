from managers.BaseManager import BaseManager
from utils import to_comb_image, t_get_confusion_matrix, t_normalise_confusion_matrix, t_get_pixel_accuracy, \
    get_matrix_fig, to_numpy, t_get_mean_iou
import torch
from torch import nn
import numpy as np
import datetime


class FCNManager(BaseManager):
    """Manager for simple img in, lbl out models"""
    def load_optimiser(self):
        """Set optimiser and if required, learning rate schedule"""
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['learning_rate'])
        if 'lr_decay_gamma' in self.config['train']:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser,
                                                                    self.config['train']['lr_decay_gamma'])

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        a = datetime.datetime.now()
        for batch_num, (img, lbl) in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            b = (datetime.datetime.now() - a).total_seconds() * 1000
            a = datetime.datetime.now()
            img, lbl = img.to(self.device), lbl.to(self.device)
            if (self.epoch + self.start_epoch) == 0 and batch_num == 0:  # To add the graph to TensorBoard
                self.model.eval()
                self.train_writer.add_graph(self.model, img.float())
                self.model.train()
            self.optimiser.zero_grad()
            output = self.model(img.float())
            if batch_num == 0:
                rec_num = 0  # Just take the first of the batch (will be random)
                lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                self.train_writer.add_image(
                    'train_images/record_{:02d}'.format(rec_num),
                    to_comb_image(img[rec_num], lbl[rec_num], lbl_pred[rec_num], self.config['data']['experiment']),
                    self.global_step, dataformats='HWC')
            loss = self.loss(output, lbl.long())
            loss.backward()
            self.optimiser.step()
            self.train_writer.add_scalar('metrics/loss', loss.item(), self.global_step)

            pa, pac = t_get_pixel_accuracy(t_get_confusion_matrix(output, lbl))
            self.train_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
            self.train_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)
            self.global_step += 1
            print("\rEpoch {:03d}, Batch {:03d} - Loss: {:.5f}; Time taken: {}"
                  .format(self.epoch + self.start_epoch, batch_num, loss.item(), b), end='', flush=True)
        if self.scheduler is not None:
            self.scheduler.step()
            self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step)

    def validate(self):
        """Validate the model on the validation data"""
        self.model.eval()
        valid_loss = 0
        confusion_matrix = None
        with torch.no_grad():
            for rec_num, (img, lbl) in enumerate(self.data_loaders['valid_loader']):
                img, lbl = img.to(self.device), lbl.to(self.device)
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
        self.valid_writer.add_scalar('metrics/loss', valid_loss, self.global_step - 1)
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
        self.valid_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step - 1)

        print("\rEpoch {:03d} - Validation loss: {:.5f} - miou:{:.2f} - miou-instruments{:.2f} -"
              " miou-anatomies{:.2f}".format(self.epoch + self.start_epoch, valid_loss, m_iou,
                                             m_iou_instruments, m_iou_anatomies))

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.save_checkpoint(is_best=True)
        elif self.epoch % self.config['log_every_n_epochs'] == 0:
            self.save_checkpoint(is_best=False)
