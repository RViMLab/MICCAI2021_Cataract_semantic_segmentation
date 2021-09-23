from managers.BaseManager import BaseManager
# noinspection PyUnresolvedReferences
from models import EncDec
from losses import LossWrapper
from utils import to_comb_image, t_get_confusion_matrix, t_normalise_confusion_matrix, t_get_pixel_accuracy, \
    get_matrix_fig, to_numpy, t_get_mean_iou, un_normalise, fig_from_dist, point_sample, CLASS_INFO
import torch
from torch import nn
import numpy as np
import cv2
import pathlib


class EncDecManager(BaseManager):
    """Manager for simple img in, lbl out models"""
    def load_model(self):
        self.model = EncDec(self.config, self.experiment)
        self.model = self.model.to(self.device)
        num_train_params = self.model.get_num_params()
        print("Using encoder '{}' and decoder {}: {} trainable parameters"
              .format(self.config['encoder']['model'], self.config['decoder']['model'], num_train_params))

    def load_loss(self):
        """Load loss function"""
        self.config['loss']['experiment'] = self.experiment
        self.config['loss']['device'] = str(self.device)
        self.loss = LossWrapper(self.config['loss'])
        self.loss = self.loss.to(self.device)
        print("Loaded loss: " + self.loss.info_string)

    def train(self):
        """Main training loop"""
        num_epochs = self.config['train']['epochs']
        self.metrics['ind_dist'] = []

        # DEBUGGING
        if self.debugging:
            if not pathlib.Path.is_dir(self.log_dir / 'imgs'):
                pathlib.Path.mkdir(self.log_dir / 'imgs')

        print("\n***** Training started *****\n")
        for self.epoch in range(num_epochs):
            self.train_one_epoch()
            self.validate()

        self.metrics['ind_dist'].append(np.sum(self.metrics['ind_dist'], axis=0))
        if 'train_adaptive_batching_loader' in self.train_schedule.values():
            self.train_writer.add_figure('index_distribution/overall'.format(self.epoch + self.start_epoch),
                                         fig_from_dist(np.arange(len(self.data_loaders['train_loader'].dataset)),
                                                       np.array(self.metrics['ind_dist'][-1]),
                                                       100, 'Dataframe index', 'Count'),
                                         self.global_step - 1)

        print("\n***** Training finished *****\n"
              "Run ID: {}\n"
              "     Best validation loss: {:.5f}\n"
              "     Best mIoU         (tot / anat / ins): {:.4f} / {:.4f} / {:.4f} / {:.4f} @ epoch {} (step {})\n"
              "     mIoU at best loss (tot / anat / ins): {:.4f} / {:.4f} / {:.4f} / {:.4f} @ epoch {} (step {})"
              .format(self.run_id, self.best_loss, self.metrics['best_miou'], self.metrics['best_miou_anatomies'],
                      self.metrics['best_miou_instruments'], self.metrics['best_miou_rare'],
                      *self.metrics['best_miou_epoch_step'], self.metrics['best_loss_miou'],
                      self.metrics['best_loss_miou_anatomies'], self.metrics['best_loss_miou_instruments'],
                      self.metrics['best_loss_miou_rare'], *self.metrics['best_loss_epoch_step']))
        self.metrics['ind_dist'] = np.array(self.metrics['ind_dist'])
        # Save 'ind_dist' in the metrics in .npz file, and delete from metrics before they are saved in 'text' in
        # Tensorboard / in the jsons
        np.savez_compressed(self.log_dir / 'ind_dist.npz', self.metrics.pop('ind_dist', None))
        self.finalise()

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        loader = self.data_loaders[self.train_schedule[self.epoch]]
        self.metrics['ind_dist'].append(np.zeros((len(loader.dataset)), 'i'))
        running_confusion_matrix = 0
        for batch_num, (img, lbl, metadata) in enumerate(loader):
            self.metrics['ind_dist'][self.epoch][metadata['index'].tolist()] += 1  # update the count
            img, lbl = img.to(self.device), lbl.to(self.device)
            if (self.epoch + self.start_epoch) == 0 and batch_num == 0:  # To add the graph to TensorBoard
                self.model.eval()
                self.train_writer.add_graph(self.model, img.float())
                self.model.train()
            self.optimiser.zero_grad()
            deep_features, output, loss = self._model_and_loss(img.float(), lbl)

            # DEBUG
            if self.debugging:
                lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                rec_num = 0
                if batch_num % 10 == 0:
                    comb_image = to_comb_image(un_normalise(img[rec_num]), lbl[rec_num], lbl_pred[rec_num],
                                               self.config['data']['experiment'])[..., ::-1]
                    cv2.imwrite(str(self.log_dir / 'imgs' / 'train_img_batch{:03d}_rec{:02d}.png'
                                    .format(batch_num, rec_num)), comb_image)

            if batch_num == 0:
                rec_num = 0  # Just take the first of the batch (will be random)
                lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                self.train_writer.add_image(
                    'train_images/record_{:02d}'.format(rec_num),
                    to_comb_image(un_normalise(img[rec_num]), lbl[rec_num], lbl_pred[rec_num],
                                  self.config['data']['experiment']),
                    self.global_step, dataformats='HWC')
            loss.backward()
            self.optimiser.step()
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
            m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare = \
                iou.mean(), iou_instruments.mean(), iou_anatomies.mean(), iou_rare.mean()
            self.train_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
            self.train_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)
            self.train_writer.add_scalar('metrics/mean_iou', m_iou, self.global_step)
            self.train_writer.add_scalar('metrics/mean_iou_anatomies', m_iou_anatomies, self.global_step)
            self.train_writer.add_scalar('metrics/mean_iou_instruments', m_iou_instruments, self.global_step)
            self.train_writer.add_scalar('metrics/mean_iou_rare', m_iou_rare, self.global_step)
            if self.config['train']['lr_batchwise']:
                self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step)
                self.scheduler.step()
            self.global_step += 1
            print(("\rEpoch {:03d}, Batch {:03d} - " + self._get_loss_string())
                  .format(self.epoch + self.start_epoch, batch_num), end='', flush=True)
        if 'DenseContrastiveLoss' in self.loss.loss_classes:
            col_confusion_matrix = t_normalise_confusion_matrix(running_confusion_matrix, mode='col')
            self.train_writer.add_figure('train_confusion_matrix/col_normalised',
                                         get_matrix_fig(to_numpy(col_confusion_matrix),
                                                        self.config['data']['experiment']),
                                         self.global_step - 1)
            self.loss.loss_classes['DenseContrastiveLoss'].update_confusion_matrix(col_confusion_matrix)
        self.train_writer.add_figure('index_distribution/per_epoch'.format(self.epoch + self.start_epoch),
                                     fig_from_dist(np.arange(len(self.data_loaders['train_loader'].dataset)),
                                                   np.array(self.metrics['ind_dist'][self.epoch]),
                                                   100, 'Dataframe index', 'Count'),
                                     self.global_step - 1)
        if not self.config['train']['lr_batchwise']:
            self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step)
            self.scheduler.step()

    def _get_loss_string(self, loss: np.ndarray = None):
        loss_string = "Loss: {:.5f}".format(self.loss.total_loss.item() if loss is None else loss[0])
        for i, k in enumerate(sorted(self.loss.loss_vals.keys())):
            if i == 0:
                loss_string += " ("
            # noinspection PyUnresolvedReferences
            loss_string += "{}: {:.5f} / ".format(k, self.loss.loss_vals[k].item() if loss is None else loss[i + 1])
            if i == len(self.loss.loss_vals.keys()) - 1:
                loss_string = loss_string[:-3]
                loss_string += ")"
        return loss_string

    def _model_and_loss(self, img, lbl):
        output = self.model(img)
        if self.config['decoder']['model'] == 'PointRend':
            # output from network is 'point_coords, point_logits, seg_logits, pred'
            deep_features, point_coords, point_logits, seg_logits, prediction = output
            loss_coarse = self.loss(deep_features, seg_logits, lbl.long(), epoch=self.epoch)
            lbl_logits = point_sample(lbl.unsqueeze(1).float(), point_coords, mode='nearest').squeeze(1)
            if self.experiment in [2, 3]:
                ignore_class = len(CLASS_INFO[self.experiment][1]) - 1
            else:
                ignore_class = -100
            ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_class)
            loss_points = ce_loss(point_logits.unsqueeze(3), lbl_logits.unsqueeze(2).long())
            loss = loss_coarse + loss_points
            self.train_writer.add_scalar('metrics/loss0_total', loss.item(), self.global_step)
            for i, (loss_name, loss_value) in enumerate(sorted(self.loss.loss_vals.items())):
                self.train_writer.add_scalar('metrics/loss{}_'.format(i) + loss_name,
                                             loss_value.item(), self.global_step)
            self.train_writer.add_scalar('metrics/loss_coarse', loss_coarse.item(), self.global_step)
            self.train_writer.add_scalar('metrics/loss_points', loss_points.item(), self.global_step)
        else:
            deep_features, prediction = output
            loss = self.loss(deep_features, prediction, lbl.long(), epoch=self.epoch)
            self.train_writer.add_scalar('metrics/loss0_total', loss.item(), self.global_step)
            for i, (loss_name, loss_value) in enumerate(sorted(self.loss.loss_vals.items())):
                self.train_writer.add_scalar('metrics/loss{}_'.format(i + 1) + loss_name,
                                             loss_value.item(), self.global_step)
        return deep_features, prediction, loss

    def validate(self):
        """Validate the model on the validation data"""
        self.model.eval()
        valid_loss = []
        confusion_matrix = None
        with torch.no_grad():
            for rec_num, (img, lbl, metadata) in enumerate(self.data_loaders['valid_loader']):
                img, lbl = img.to(self.device), lbl.to(self.device)
                deep_features, output = self.model(img.float())
                loss = self.loss(deep_features, output, lbl.long(), epoch=self.epoch).item()
                valid_loss.append([loss,
                                   *[self.loss.loss_vals[k].item() for k in sorted(self.loss.loss_vals.keys())]])
                confusion_matrix = t_get_confusion_matrix(output, lbl, confusion_matrix)

                # DEBUG
                if self.debugging:
                    lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                    comb_image = to_comb_image(un_normalise(img[0]), lbl[0], lbl_pred[0],
                                               self.config['data']['experiment'])[..., ::-1]
                    cv2.imwrite(str(self.log_dir / 'imgs' / 'valid_img_batch{:04d}.png'.format(rec_num)), comb_image)

                if rec_num in np.round(np.linspace(0, len(self.data_loaders['valid_loader']) - 1, self.max_valid_imgs)):
                    # perf = calculate_performance(output, lbl, metadata)
                    # for index, perf_tensor in perf.items():
                    #     perf_img = colourise_data(to_numpy(perf_tensor), low=0, high=1)
                    #     self.valid_writer.add_image('valid_performance/record_{:02d}_ind{:06d}'
                    #                                 .format(rec_num, index),
                    #                                 perf_img[0], self.global_step - 1, dataformats='HWC')
                    lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                    self.valid_writer.add_image(
                        'valid_images/record_{:02d}'.format(rec_num),
                        to_comb_image(un_normalise(img[0]), lbl[0], lbl_pred[0], self.config['data']['experiment']),
                        self.global_step, dataformats='HWC')
        valid_loss = np.mean(valid_loss, axis=0)
        row_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'row')
        col_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'col')
        self.valid_writer.add_figure('valid_confusion_matrix/row_normalised',
                                     get_matrix_fig(to_numpy(row_confusion_matrix), self.config['data']['experiment']),
                                     self.global_step - 1)
        self.valid_writer.add_figure('valid_confusion_matrix/col_normalised',
                                     get_matrix_fig(to_numpy(col_confusion_matrix), self.config['data']['experiment']),
                                     self.global_step - 1)
        pa, pac = t_get_pixel_accuracy(confusion_matrix)
        m_iou, m_iou_instruments, m_iou_anatomies, m_iou_rare = \
            t_get_mean_iou(confusion_matrix, self.config['data']['experiment'], categories=True, rare=True)
        self.valid_writer.add_scalar('metrics/loss0_total', valid_loss[0], self.global_step - 1)
        for i, (loss_name, loss_value) in enumerate(zip(sorted(self.loss.loss_vals.keys()), valid_loss[1:])):
            self.valid_writer.add_scalar('metrics/loss{}_'.format(i + 1) + loss_name,
                                         loss_value.item(), self.global_step - 1)
        self.valid_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/mean_iou', m_iou, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/mean_iou_anatomies', m_iou_anatomies, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/mean_iou_instruments', m_iou_instruments, self.global_step - 1)
        self.valid_writer.add_scalar('metrics/mean_iou_rare', m_iou_rare, self.global_step - 1)
        print(("\rEpoch {:03d} - Validation " + self._get_loss_string(valid_loss))
              .format(self.epoch + self.start_epoch))
        best_miou_flag = False
        if m_iou > self.metrics['best_miou']:
            self.metrics.update({'best_miou': m_iou,
                                 'best_miou_anatomies': m_iou_anatomies,
                                 'best_miou_instruments': m_iou_instruments,
                                 'best_miou_rare': m_iou_rare,
                                 'best_miou_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            best_miou_flag = True
        if valid_loss[-1] < self.best_loss:
            self.best_loss = valid_loss[-1]
            self.metrics.update({'best_loss_miou': m_iou,
                                 'best_loss_miou_anatomies': m_iou_anatomies,
                                 'best_loss_miou_instruments': m_iou_instruments,
                                 'best_loss_miou_rare': m_iou_rare,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            print("            New best validation loss: {:.5f}".format(valid_loss[-1]))
            if best_miou_flag:
                print("            --- with new best mIoU (tot / anat / ins / rare): {:.4f} / {:.4f} / {:.4f} / {:.4f}"
                      .format(m_iou, m_iou_anatomies, m_iou_instruments, m_iou_rare))
            else:
                print("            --- with mIoU (tot / anat / ins / rare): {:.4f} / {:.4f} / {:.4f} / {:.4f}\n"
                      "            --- best mIoU (tot / anat / ins / rare): {:.4f} / {:.4f} / {:.4f} / {:.4f}"
                      .format(m_iou, m_iou_anatomies, m_iou_instruments, m_iou_rare, self.metrics['best_miou'],
                              self.metrics['best_miou_anatomies'], self.metrics['best_miou_instruments'],
                              self.metrics['best_miou_rare']))
            self.save_checkpoint(is_best=True)
        elif self.epoch % self.config['log_every_n_epochs'] == 0:
            self.save_checkpoint(is_best=False)

        # Update info.json file so it exists in case the run stops / crashes before self.finalise()
        self.write_info_json()
