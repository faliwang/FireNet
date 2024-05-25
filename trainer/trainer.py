import collections
import numpy as np
import torch
from abc import abstractmethod
from numpy import inf
from logger.logger import TensorboardWriter
# local modules
from utils.util import mean, inf_loop, MetricTracker
from utils.training_utils import make_flow_movie, select_evenly_spaced_elements, make_tc_vis, make_vw_vis
from utils.data import data_sources


class Trainer:
    """
    Trainer class
    """
    def __init__(self, model, loss_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_ftns = loss_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume,
                    reset_monitor_best=cfg_trainer.get('reset_monitor_best', False))
        
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(len(data_loader) // 100, 1)
        self.val_log_step = max(len(valid_data_loader) // 100, 1)

        mt_keys = ['loss']
        for data_source in data_sources:
            mt_keys.append(f'loss/{data_source}')
            for l in self.loss_ftns:
                mt_keys.append(f'{l.__class__.__name__}/{data_source}')
        self.train_metrics = MetricTracker(*mt_keys, writer=self.writer)
        self.valid_metrics = MetricTracker(*mt_keys, writer=self.writer)

        self.num_previews = config['trainer']['num_previews']
        self.val_num_previews = config['trainer'].get('val_num_previews', self.num_previews)
        self.val_preview_indices = select_evenly_spaced_elements(self.val_num_previews, len(self.valid_data_loader))
        self.valid_only = config['trainer'].get('valid_only', False)
        self.true_once = True  # True at init, turns False at end of _train_epoch

    def to_device(self, item):
        events = item['events'].float().to(self.device)
        image = item['frame'].float().to(self.device)
        flow = None if item['flow'] is None else item['flow'].float().to(self.device)
        return events, image, flow

    def forward_sequence(self, sequence, all_losses=False):
        losses = collections.defaultdict(list)
        self.model_states = None
        for i, item in enumerate(sequence):
            events, image, flow = self.to_device(item)
            pred = self.model(events, self.model_states)
            self.model_states = pred['state']
            for loss_ftn in self.loss_ftns:
                loss_name = loss_ftn.__class__.__name__
                tmp_weight = loss_ftn.weight 
                if all_losses:
                    loss_ftn.weight = 1.0
                if loss_name == 'perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], image, normalize=True))
                if loss_name == 'l2_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], image))
                if loss_name == 'temporal_consistency_loss':
                    l = loss_ftn(i, image, pred['image'], flow)
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name in ['flow_loss', 'flow_l1_loss'] and flow is not None:
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'warping_flow_loss':
                    l = loss_ftn(i, image, pred['flow'])
                    if l is not None:
                        losses[loss_name].append(l)
                if loss_name == 'voxel_warp_flow_loss' and flow is not None:
                    losses[loss_name].append(loss_ftn(events, pred['flow']))
                if loss_name == 'flow_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['flow'], flow))
                if loss_name == 'combined_perceptual_loss':
                    losses[loss_name].append(loss_ftn(pred['image'], pred['flow'], image, flow))
                loss_ftn.weight = tmp_weight
        idx = int(item['data_source_idx'].mode().values.item())
        data_source = data_sources[idx]
        losses = {f'{k}/{data_source}': mean(v) for k, v in losses.items()}
        losses['loss'] = sum(losses.values())
        losses[f'loss/{data_source}'] = losses['loss']
        return losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.valid_only:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                return {'val_' + k : v for k, v in val_log.items()}
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for k, v in losses.items():
                self.train_metrics.update(k, v.item())

            if batch_idx % self.log_step == 0:
                msg = 'Train Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx < self.num_previews and (epoch - 1) % self.save_period == 0:
                with torch.no_grad():
                    self.preview(sequence, epoch, tag_prefix=f'train_{batch_idx}')

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        print("validation")
        if self.do_validation and epoch%10==0:
            with torch.no_grad():
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_' + k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.true_once = False
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        i = 0
        for batch_idx, sequence in enumerate(self.valid_data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_sequence(sequence, all_losses=True)
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            for k, v in losses.items():
                self.valid_metrics.update(k, v.item())

            if batch_idx % self.val_log_step == 0:
                msg = 'Valid Epoch: {} {}'.format(epoch, self._progress(batch_idx, self.valid_data_loader))
                for k, v in losses.items():
                    msg += ' {}: {:.4f}'.format(k[:4], v.item())
                self.logger.debug(msg)

            if batch_idx in self.val_preview_indices and (epoch - 1) % self.save_period == 0:
                self.preview(sequence, epoch, tag_prefix=f'val_{i}')
                i += 1

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len(data_loader)
        return base.format(current, total, 100.0 * current / total)

    def preview(self, sequence, epoch, tag_prefix=''):
        """
        Plot visualisation to tensorboard.
        Plots input, output, groundtruth histograms and movies
        """
        print(f'Making preview {tag_prefix}')
        event_previews, pred_flows, pred_images, flows, images, voxels = [], [], [], [], [], []
        self.model_states = None
        for i, item in enumerate(sequence):
            item = {k: v[0:1, ...] for k, v in item.items()}  # set batch size to 1
            events, image, flow = self.to_device(item)
            pred = self.model(events, self.model_states)
            self.model_states = pred['state']
            event_previews.append(torch.sum(events, dim=1, keepdim=True))
            pred_flows.append(pred.get('flow', 0 * flow))
            pred_images.append(pred['image'])
            flows.append(flow)
            images.append(image)
            voxels.append(events)

        tc_loss_ftn = self.get_loss_ftn('temporal_consistency_loss')
        if self.true_once and tc_loss_ftn is not None:
            for i, image in enumerate(images):
                output = tc_loss_ftn(i, image, pred_images[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_tc_vis(output[1])
                    self.writer.writer.add_video(f'warp_vis/tc_{tag_prefix}',
                            video_tensor, global_step=epoch, fps=2)
                    break

        vw_loss_ftn = self.get_loss_ftn('voxel_warp_flow_loss')
        if self.true_once and vw_loss_ftn is not None:
            for i, image in enumerate(images):
                output = vw_loss_ftn(voxels[i], flows[i], output_images=True)
                if output is not None:
                    video_tensor = make_vw_vis(output[1])
                    self.writer.writer.add_video(f'warp_vox/tc_{tag_prefix}',
                            video_tensor, global_step=epoch, fps=1)
                    break
        
        non_zero_voxel = torch.stack([s['events'] for s in sequence])
        non_zero_voxel = non_zero_voxel[non_zero_voxel != 0]
        if torch.numel(non_zero_voxel) == 0:
            non_zero_voxel = 0
        self.writer.add_histogram(f'{tag_prefix}_flow/groundtruth',
                                  torch.stack(flows))
        self.writer.add_histogram(f'{tag_prefix}_image/groundtruth',
                                  torch.stack(images))
        self.writer.add_histogram(f'{tag_prefix}_input',
                                  non_zero_voxel)
        self.writer.add_histogram(f'{tag_prefix}_flow/prediction',
                                  torch.stack(pred_flows))
        self.writer.add_histogram(f'{tag_prefix}_image/prediction',
                                  torch.stack(pred_images))
        video_tensor = make_flow_movie(event_previews, pred_images, images, pred_flows, flows)
        self.writer.writer.add_video(f'{tag_prefix}', video_tensor, global_step=epoch, fps=20)

    def get_loss_ftn(self, loss_name):
        for loss_ftn in self.loss_ftns:
            if loss_ftn.__class__.__name__ == loss_name:
                return loss_ftn
        return None
    
    def add_dict(self, result, epoch):
        for k, v in result.items():
            if v == 0:
                continue
            if '/' not in k:
                k += '_total'
            if 'val_' in k:
                k = f'{k[4:]}/valid'
            else:
                k += '/train'
            self.writer.writer.add_scalar(f'epoch_{k}', v, global_step=epoch)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            self.add_dict(result, epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, reset_monitor_best=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        if not reset_monitor_best:
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))