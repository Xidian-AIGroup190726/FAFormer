import mmcv
from logging import Logger
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Optimizer, Adam, RMSprop, SGD, AdamW, lr_scheduler
import os.path as osp
import cv2
import numpy as np

from datasets.utils import save_image, data_augmentation
from .common.utils import torch2np, dwt2d, smart_time, set_batch_cuda
from .common.losses import get_loss_module
from .common import metrics as mtc
from torch.utils.tensorboard import SummaryWriter


class Base_model:
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        r"""
        Args:
            cfg (mmcv.Config): full config
            logger (Logger)
            train_data_loader (data.DataLoader): dataloader for training
            test_data_loader0 (data.DataLoader): dataloader for full-resolution testing
            test_data_loader1 (data.DataLoader): dataloader for low-resolution testing
        """
        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.logger = logger
        self.train_data_loader = train_data_loader
        self.test_data_loader0 = test_data_loader0
        self.test_data_loader1 = test_data_loader1

        mmcv.mkdir_or_exist(self.work_dir)
        self.train_out = f'{self.work_dir}/train_out'
        self.test_out0 = f'{self.work_dir}/test_out0'
        self.test_out1 = f'{self.work_dir}/test_out1'

        self.eval_results = {}
        self.module_dict = {}
        self.optim_dict = {}
        self.sched_dict = {}
        self.switch_dict = {}
        self.loss_module = get_loss_module(full_cfg=cfg, logger=logger)
        self.last_iter = 0

        self.writer = SummaryWriter()

    def showData(self, value, epoch):
        for name in value:
            target = None
            if name.find('loss') != -1:
                target = 'Loss/' + name
            elif name.find('mean') != -1:
                target = 'metric_mean/' + name

            if target is not None:
                if isinstance(value[name], float):
                    self.writer.add_scalar(target, value[name], epoch)
                else:
                    for item in (value[name]):
                        self.writer.add_scalar(target, item, epoch)

    def add_module(self, module_name: str, module: nn.Module, switch=True):
        assert isinstance(module, nn.Module)
        self.module_dict[module_name] = module
        self.switch_dict[module_name] = switch

    def add_optim(self, optim_name: str, optim: Optimizer):
        self.optim_dict[optim_name] = optim

    def add_sched(self, sched_name: str, sched: lr_scheduler.StepLR):
        self.sched_dict[sched_name] = sched

    def print_total_params(self):
        count = 0
        for module_name in self.module_dict:
            module = self.module_dict[module_name]
            param_num = sum(p.numel() for p in module.parameters())
            self.logger.info(f'total params of "{module_name}": {param_num}')
            count += param_num
        self.logger.info(f'total params: {count}')

    def print_total_trainable_params(self):
        count = 0
        for module_name in self.module_dict:
            module = self.module_dict[module_name]
            param_num = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.logger.info(f'total trainable params of "{module_name}": {param_num}')
            count += param_num
        self.logger.info(f'total trainable params: {count}')

    def init(self):
        pass
        # for module in self.module_dict.values():
        #    module.apply(weight_init)

    def set_cuda(self):
        for module_name in self.module_dict:
            self.logger.debug(module_name)
            module = self.module_dict[module_name]
            if torch.cuda.device_count() > 1:
                module = nn.DataParallel(module)
            self.module_dict[module_name] = module.cuda()
        for loss_name in self.loss_module:
            loss = self.loss_module[loss_name]
            self.loss_module[loss_name] = loss.cuda()

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.last_iter = checkpoint["iter_num"]
        self.logger.debug(f'last_iter: {self.last_iter}')
        for module_name in self.module_dict:
            module = self.module_dict[module_name]
            module.load_state_dict(checkpoint[module_name].state_dict())

    def load_pretrained(self, path: str):
        checkpoint = torch.load(path)
        for module_name in self.module_dict:
            module = self.module_dict[module_name]
            module.load_state_dict(checkpoint[module_name].state_dict())

    # def load_pretrained(self, path: str):
    #     checkpoint = torch.load(path)
    #     ck_state_dict = checkpoint['core_module']
    #
    #     for module_name in self.module_dict:
    #         module = self.module_dict[module_name]
    #         load_info = module.load_state_dict(ck_state_dict, strict=False)
    #
    #         print("Missing keys:", load_info.missing_keys)  # 'ms_KV_cross_attn.0.layers.0.0.attention_block.fn.fn.iab.alpha.bottleneckBlock.0.weight'
    #         print("Unexpected keys:", load_info.unexpected_keys)
    #         num_loaded = len(ck_state_dict) - len(load_info.unexpected_keys)
    #         print("Successfully loaded params:", num_loaded)
    #         print("Total in ck_state_dict:", len(ck_state_dict))
    #         print("Total missing in module:", len(load_info.missing_keys))

    def set_optim(self):
        optim_cfg = self.cfg.get('optim_cfg', {})
        for module_name in self.module_dict:
            module = self.module_dict[module_name]
            if module_name in optim_cfg:
                cfg = optim_cfg[module_name].copy()
                _cfg = cfg.copy()
                _cfg.pop('type')
                if cfg.type == 'Adam':
                    self.optim_dict[module_name] = Adam(module.parameters(), **_cfg)
                elif cfg.type == 'RMSprop':
                    self.optim_dict[module_name] = RMSprop(module.parameters(), **_cfg)
                elif cfg.type == 'SGD':
                    self.optim_dict[module_name] = SGD(module.parameters(), **_cfg)
                elif cfg.type == 'AdamW':
                    self.optim_dict[module_name] = AdamW(module.parameters(), **_cfg)
                else:
                    raise SystemExit(f'No such type optim:{cfg.type}')
            else:
                self.optim_dict[module_name] = Adam(module.parameters(), betas=(0.9, 0.999), lr=1e-4)

    def set_sched(self):
        # decay the learning rate every n 'iterations'
        sched_cfg = self.cfg.get('sched_cfg', dict(step_size=10000, gamma=0.99))
        for optim_name in self.optim_dict:
            optim = self.optim_dict[optim_name]
            self.sched_dict[optim_name] = lr_scheduler.StepLR(
                optimizer=optim, **sched_cfg
            )

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    def train(self):
        for freq_str in ['save_freq', 'test_freq', 'eval_freq']:
            self.cfg.setdefault(freq_str, 10000)
        self.cfg.setdefault('max_iter', 100000)

        self.before_train()
        self.timer = mmcv.Timer()
        iter_id = self.last_iter
        while iter_id < self.cfg.max_iter:
            for _, input_batch in enumerate(self.train_data_loader):
                if self.cfg.cuda:
                    input_batch = set_batch_cuda(input_batch)
                if 'aug_dict' in self.cfg:
                    input_batch = data_augmentation(input_batch, self.cfg.aug_dict)
                iter_id += 1
                for module in self.module_dict.values():
                    module.train()
                self.before_train_iter()
                loss_res = self.train_iter(iter_id=iter_id, input_batch=input_batch)
                self.showData(loss_res, iter_id)
                self.after_train_iter()

                def should(freq):
                    return (freq != -1) and (iter_id % freq == 0) and (iter_id != self.cfg.max_iter)

                if should(self.cfg.save_freq):
                    self.save(iter_id=iter_id)
                if should(self.cfg.eval_freq):
                    # self.test(iter_id=iter_id, save=should(self.cfg.test_freq), ref=False)
                    self.test(iter_id=iter_id, save=should(self.cfg.test_freq), ref=True)
                for sched_name in self.sched_dict:
                    if self.switch_dict[sched_name]:
                        self.sched_dict[sched_name].step()

                if iter_id == self.cfg.max_iter:
                    break

        self.after_train()

    def train_iter(self, iter_id, input_batch, log_freq=100):

        # 初始化各模型
        Core = self.module_dict['core_module']
        Core_optim = self.optim_dict['core_module']

        loss = 0.0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})

        # 模型运行
        # 初始化各类型的遥感图像
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']

        # 模型运行
        output, l_dwt_ms, h_dwt_pan = Core(input_lr, input_pan)

        # 计算损失函数
        if 'QNR_loss' in self.loss_module and loss_cfg['QNR_loss'].w != 0.0 :
            input_pan_l = input_batch['input_pan_l']
            QNR_loss = self.loss_module['QNR_loss'](pan=input_pan, ms=input_lr, pan_l=input_pan_l, out=output)
            loss += QNR_loss * loss_cfg['QNR_loss'].w
            loss_res['QNR_loss'] = QNR_loss.item()

        l_out, h_out = dwt2d(output)
        target = input_batch['target']
        l_gt, h_gt = dwt2d(target)

        if 'spectral_rec_loss' in self.loss_module and loss_cfg['spectral_rec_loss'].w != 0.0:
            spectral_rec_loss = self.loss_module['spectral_rec_loss'](
                out=l_out, gt=l_gt)
            loss += spectral_rec_loss * loss_cfg['spectral_rec_loss'].w
            loss_res['spectral_rec_loss'] = spectral_rec_loss.item()

        if 'spatial_rec_loss' in self.loss_module and loss_cfg['spatial_rec_loss'].w != 0.0:
            spatial_rec_loss = self.loss_module['spatial_rec_loss'](
                out=h_out, gt=h_gt)
            loss += spatial_rec_loss * loss_cfg['spatial_rec_loss'].w
            loss_res['spatial_rec_loss'] = spatial_rec_loss.item()

        if 'rec_loss' in self.loss_module and loss_cfg['rec_loss'].w != 0.0:
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target)
            loss += rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()

        loss_res['full_loss'] = loss.item()

        # 更新参数
        Core_optim.zero_grad()
        loss.backward()
        Core_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)

        return loss_res

    def print_train_log(self, iter_id, loss_res, log_freq=10):
        r""" print current loss at one iteration

        Args:
            iter_id (int)
            loss_res (dict[str, float | tuple[float]]
            log_freq (int)
        """
        if iter_id % log_freq == 0:
            avg_iter_time = self.timer.since_last_check() / log_freq
            remain_time = avg_iter_time * (self.cfg.max_iter - iter_id)
            self.logger.info(f'===> training iteration[{iter_id}/{self.cfg.max_iter}] '
                             f'lr: {self.optim_dict["core_module"].param_groups[0]["lr"]:.6f}, '
                             f'ETA: {smart_time(remain_time)}')
            for loss_name in loss_res:
                if 'rec_loss' in loss_name:
                    self.logger.info(f'{loss_name}_{self.loss_module[loss_name].get_type()}: '
                                     f'{loss_res[loss_name]:.6f}')

                if 'QNR_loss' in loss_name:
                    self.logger.info(f'QNR_loss: {loss_res[loss_name]:.6f}')

                if 'full_loss' in loss_name:
                    self.logger.info(f'full_loss: {loss_res[loss_name]:.6f}')

    def test(self, iter_id, save, ref):
        r""" test and evaluate the model

        Args:
            iter_id (int): current iteration num
            save (bool): whether to save the output of test images
            ref (bool): True for low-res testing, False for full-res testing
        """
        use_sewar = self.cfg.get('use_sewar', False)
        self.logger.info(f'{"Low" if ref else "Full"} resolution testing {"with sewar" if use_sewar else ""}...')
        for module in self.module_dict.values():
            module.eval()

        test_path = osp.join(self.test_out1 if ref else self.test_out0, f'iter_{iter_id}')
        if save:
            mmcv.mkdir_or_exist(test_path)

        tot_time = 0
        tot_count = 0
        tmp_results = {}
        eval_metrics = ['SAM', 'ERGAS', 'Q4', 'SCC', 'SSIM', 'MPSNR'] if ref \
            else ['D_lambda', 'D_s', 'QNR', 'FCC', 'SF', 'SD', 'SAM_nrf']
        for metric in eval_metrics:
            tmp_results.setdefault(metric, [])

        for _, input_batch in enumerate(self.test_data_loader1 if ref else self.test_data_loader0):
            if self.cfg.cuda:
                input_batch = set_batch_cuda(input_batch)
            image_ids = input_batch['image_id']
            n = len(image_ids)
            tot_count += n

            Core = self.module_dict['core_module']

            timer = mmcv.Timer()
            with torch.no_grad():
                # 模型运行
                # 初始化各类型的遥感图像
                input_pan = input_batch['input_pan']
                input_lr = input_batch['input_lr']

                # 模型运行
                output, _, _ = Core(input_lr, input_pan)
            tot_time += timer.since_start()

            input_pan = torch2np(input_batch['input_pan'])  # shape of [N, H, W]
            input_lr = torch2np(input_batch['input_lr'])  # shape of [N, H, W, C]
            if ref:
                target = torch2np(input_batch['target'])
            output_np = torch2np(output)

            for i in range(n):

                if ref:
                    tmp_results['SAM'].append(mtc.SAM_numpy(target[i], output_np[i], sewar=use_sewar))
                    tmp_results['ERGAS'].append(mtc.ERGAS_numpy(target[i], output_np[i], sewar=use_sewar))
                    if self.cfg.ms_chans == 4:
                        tmp_results['Q4'].append(mtc.Q4_numpy(target[i], output_np[i]))
                    tmp_results['SCC'].append(mtc.SCC_numpy(target[i], output_np[i], sewar=use_sewar))
                    tmp_results['SSIM'].append(mtc.SSIM_numpy(target[i], output_np[i], 2 ** self.cfg.bit_depth - 1,
                                                              sewar=use_sewar))
                    tmp_results['MPSNR'].append(mtc.MPSNR_numpy(target[i], output_np[i], 2 ** self.cfg.bit_depth - 1))
                else:
                    tmp_results['D_lambda'].append(mtc.D_lambda_numpy(input_lr[i], output_np[i], sewar=use_sewar))
                    tmp_results['D_s'].append(mtc.D_s_numpy(input_lr[i], input_pan[i], output_np[i], sewar=use_sewar))
                    tmp_results['QNR'].append((1 - tmp_results['D_lambda'][-1]) * (1 - tmp_results['D_s'][-1]))
                    tmp_results['FCC'].append(mtc.FCC_numpy(input_pan[i], output_np[i]))
                    tmp_results['SF'].append(mtc.SF_numpy(output_np[i]))
                    tmp_results['SD'].append(mtc.SD_numpy(output_np[i]))
                    tmp_results['SAM_nrf'].append(mtc.SAM_numpy(input_lr[i], cv2.resize(output_np[i], (100, 100)), sewar=use_sewar))

                if save:
                    save_image(osp.join(test_path, f'{image_ids[i]}_mul_hat.tif'), output[i].cpu().detach().numpy())

        for metric in eval_metrics:
            self.eval_results.setdefault(f'{metric}_mean', [])
            self.eval_results.setdefault(f'{metric}_std', [])
            mean = np.mean(tmp_results[metric])
            std = np.std(tmp_results[metric])
            self.eval_results[f'{metric}_mean'].append(round(mean, 4))
            self.eval_results[f'{metric}_std'].append(round(std, 4))
            self.logger.info(f'{metric} metric value: {mean:.4f} +- {std:.4f}')

        if iter_id == self.cfg.max_iter:  # final testing
            for metric in eval_metrics:
                mean_array = self.eval_results[f'{metric}_mean']
                self.logger.info(f'{metric} metric curve: {mean_array}')
        self.logger.info(f'Avg time cost per img: {tot_time / tot_count:.5f}s')

        self.showData(self.eval_results, iter_id)

    def save(self, iter_id):
        r""" save the weights of model to checkpoint

        Args:
            iter_id (int): current iteration num
        """
        mmcv.mkdir_or_exist(self.train_out)
        model_out_path = osp.join(self.train_out, f'model_iter_{iter_id}.pth')
        state = self.module_dict.copy()
        if torch.cuda.device_count() > 1:
            for module_name in self.module_dict:
                module = self.module_dict[module_name]
                state[module_name] = module.module
        state['iter_num'] = iter_id
        torch.save(state, model_out_path)
        self.logger.info("Checkpoint saved to {}".format(model_out_path))
