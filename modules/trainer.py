import logging
import os, time
import pandas as pd
from abc import abstractmethod
from modules.loss import compute_nll_sum_and_tokens

import torch
from torch.amp import GradScaler, autocast
from numpy import inf

# import wandb and handle exception
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        # ---- W&B init (1 lần) ----
        # Read from args (via .sh file) instead of env var
        self.use_wandb = WANDB_OK and not getattr(args, 'no_wandb', False)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if self.use_wandb:
            wandb.init(
                project = getattr(args, 'wandb_project', 'A3Net-IU-Xray'),
                entity = getattr(args, 'wandb_entity', None),
                name = getattr(args, 'wandb_name', None) or f"{args.dataset_name}_seed{args.seed}",
                config  = vars(args),
                resume = "allow", #  tự resume nếu đang chạy mà bị ngắt giữa chừng
                # Old
                # project=os.environ.get("WANDB_PROJECT", "A3Net-IU-Xray"),
                # name=os.environ.get("WANDB_NAME", f"{args.dataset_name}_seed{args.seed}"),
                # config=vars(args),
            )
            self.logger.info(f"WandB run: {wandb.run.url}")
        # --------------------------------------------------

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # AMP scaler: chỉ kích hoạt khi --use_amp được truyền vào
        self.use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            self.logger.info("AMP (Automatic Mixed Precision) ENABLED - VRAM usage ~halved")

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # --- Loss history để vẽ biểu đồ ----
        self.train_loss_history = []
        self.val_loss_history = []
        self.stopped_epoch = None
        self.visual_unfrozen = getattr(args, 'visual_unfreeze_epoch', 0) <= 0

        #Resume...
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    def _get_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _normalize_state_dict(self, state_dict):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            normalized[key] = value
        return normalized

    def _load_model_state(self, state_dict):
        incompatible = self._get_model().load_state_dict(
            self._normalize_state_dict(state_dict),
            strict=False
        )
        if incompatible.missing_keys:
            self.logger.warning("Missing model keys when loading checkpoint: {}".format(incompatible.missing_keys))
        unexpected_keys = [
            key for key in incompatible.unexpected_keys
            if 'fc_memory_proj' not in key and 'global_memory_scale' not in key
        ]
        if unexpected_keys:
            self.logger.warning("Unexpected model keys when loading checkpoint: {}".format(unexpected_keys))

    def _set_visual_layers_trainable(self, layer_names):
        model = self._get_model()
        visual_model = model.visual_extractor.model
        layer_name_to_index = {
            'conv1': 0,
            'bn1': 1,
            'layer1': 4,
            'layer2': 5,
            'layer3': 6,
            'layer4': 7,
        }

        if 'all' in layer_names:
            for param in model.visual_extractor.parameters():
                param.requires_grad = True
            return ['all']

        unknown_layers = set(layer_names) - set(layer_name_to_index)
        if unknown_layers:
            raise ValueError(f'Unknown visual_unfreeze_layers: {sorted(unknown_layers)}')

        for layer_name in layer_names:
            for param in visual_model[layer_name_to_index[layer_name]].parameters():
                param.requires_grad = True
        return sorted(layer_names)

    def _maybe_unfreeze_visual_extractor(self, epoch):
        unfreeze_epoch = getattr(self.args, 'visual_unfreeze_epoch', 0)
        if self.visual_unfrozen or unfreeze_epoch <= 0 or epoch < unfreeze_epoch:
            return

        layer_names = [
            name.strip()
            for name in getattr(self.args, 'visual_unfreeze_layers', 'all').split(',')
            if name.strip()
        ]
        if not layer_names:
            layer_names = ['all']

        unfrozen_layers = self._set_visual_layers_trainable(layer_names)
        self.visual_unfrozen = True
        self.logger.info(
            'Unfroze visual extractor layers at epoch {}: {}'.format(epoch, unfrozen_layers)
        )

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._maybe_unfreeze_visual_extractor(epoch)
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            log['visual_unfrozen'] = int(self.visual_unfrozen)
            self._record_best(log)

            # --- Lưu loss history để vẽ biểu đồ ---
            if 'train_loss' in log:
                self.train_loss_history.append((epoch, log['train_loss']))
            if 'val_loss' in log:
                self.val_loss_history.append((epoch, log['val_loss']))

            # log metrics to wandb
            if self.use_wandb:
                wandb.log(log, step=epoch)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    self.stopped_epoch = epoch # ghi lại epoch bị dừng sớm
                    break

            # Save best model weights
            if best:
                self._save_checkpoint(epoch, save_best=True)
            # Save checkpoint every save_period epochs
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        
        self._print_best()
        self._print_best_to_file()

        # ---------------------------------------------------------------------------
        # Finish WandB run
        # ---------------------------------------------------------------------------
        if self.use_wandb:
            wandb.finish()  
        # ---------------------------------------------------------------------------

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']]), pd.DataFrame([self.best_recorder['test']])] ,ignore_index=True)
        
        record_table.to_csv(record_path, index=False)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self._get_model().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor_best': self.mnt_best
        }

        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

            # --- Upload model_best lên WandB để team có thể download ------
            if self.use_wandb:
                wandb.save(best_path, base_path=self.checkpoint_dir)
                self.logger.info("Uploaded model_best.pth to WandB")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self._load_model_state(checkpoint['state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as exc:
            self.logger.warning("Optimizer state skipped because model parameters changed: {}".format(exc))
        if self.lr_scheduler and checkpoint.get('lr_scheduler') is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except ValueError as exc:
                self.logger.warning("LR scheduler state skipped: {}".format(exc))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        train_nll_sum = 0.0
        train_token_count = 0.0
        accum_steps = getattr(self.args, 'accum_steps', 1)  # gradient accumulation

        self.model.train()
        self.optimizer.zero_grad()  # reset gradient trước epoch
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images = images.to(self.device, non_blocking=True)
            reports_ids = reports_ids.to(self.device, non_blocking=True)
            reports_masks = reports_masks.to(self.device, non_blocking=True)

            # --- Forward (with AMP nếu được bật) ---
            with autocast('cuda', enabled=self.use_amp):
                if epoch >= getattr(self.args, 'scst_start_epoch', 20):
                    self.model.eval()
                    with torch.no_grad():
                        greedy_res, _ = self.model(images, mode='sample', update_opts={'sample_method': 'greedy'})
                    self.model.train()
                    sample_res, sample_logprobs = self.model(images, mode='sample', update_opts={'sample_method': 'sample'})
                    
                    from modules.reward import get_self_critical_reward
                    reward_type = getattr(self.args, 'scst_reward', 'cider')
                    reward, _ = get_self_critical_reward(greedy_res, sample_res, reports_ids, self._get_model().tokenizer, reward_type=reward_type)
                    reward = reward.to(self.device)
                    
                    mask = (sample_res > 0).float()
                    # Gather the log probability of the sampled token at each time step
                    # sample_res is (B, seq_len), sample_logprobs is (B, seq_len, vocab_size)
                    sample_logprobs = sample_logprobs.gather(2, sample_res.unsqueeze(2)).squeeze(2)
                    sample_logprobs = sample_logprobs * mask
                    scst_loss = - (reward * sample_logprobs.sum(1) / mask.sum(1)).mean()
                    
                    output = self.model(images, reports_ids, mode='train')
                    ce_loss = self.criterion(output, reports_ids, reports_masks)
                    
                    rl_weight = getattr(self.args, 'rl_weight', 0.99)
                    loss = rl_weight * scst_loss + (1.0 - rl_weight) * ce_loss
                else:
                    output = self.model(images, reports_ids, mode='train')
                    loss = self.criterion(output, reports_ids, reports_masks)

            # --- Backward (gradient accumulation) ---
            if self.use_amp:
                self.scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss += loss.item()

            with torch.no_grad():
                if epoch < getattr(self.args, 'scst_start_epoch', 20):
                    batch_nll_sum, batch_token_count = compute_nll_sum_and_tokens(
                        output.detach(), reports_ids, reports_masks
                    )
                    train_nll_sum += batch_nll_sum.item()
                    train_token_count += batch_token_count.item()

            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))

        log = {
            'train_loss': train_nll_sum / max(train_token_count, 1.0)
        }
        self.logger.info('[{}/{}] Training Loss: {:.5f}'.format(epoch, self.epochs, log['train_loss']))

        
        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():

            # --- Tính val_loss chuẩn: global token-level mean NLL ---
            val_nll_sum = 0.0
            val_token_count = 0.0

            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images = images.to(self.device, non_blocking=True)
                reports_ids = reports_ids.to(self.device, non_blocking=True)
                reports_masks = reports_masks.to(self.device, non_blocking=True)

                # teacher-forcing validation
                with autocast('cuda', enabled=self.use_amp):
                    output = self.model(images, reports_ids, mode='train')

                batch_nll_sum, batch_token_count = compute_nll_sum_and_tokens(
                    output, reports_ids, reports_masks
                )

                val_nll_sum += batch_nll_sum.item()
                val_token_count += batch_token_count.item()

            log['val_loss'] = val_nll_sum / max(val_token_count, 1.0)
            print('val_loss: ', log['val_loss'])

            model_core = self._get_model()
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images = images.to(self.device, non_blocking=True)
                reports_ids = reports_ids.to(self.device, non_blocking=True)
                reports_masks = reports_masks.to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp):
                    output, _ = self.model(images, mode='sample')

                reports = model_core.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model_core.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)}
            )

            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images = images.to(self.device, non_blocking=True)
                reports_ids = reports_ids.to(self.device, non_blocking=True)
                reports_masks = reports_masks.to(self.device, non_blocking=True)
                with autocast('cuda', enabled=self.use_amp):
                    output, _ = self.model(images, mode='sample')

                reports = model_core.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model_core.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)}
            )

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            plateau_metric = log.get(self.mnt_metric, log.get('val_loss'))
            if plateau_metric is None:
                self.logger.warning(
                    'ReduceLROnPlateau skipped: neither {} nor val_loss is available.'.format(self.mnt_metric)
                )
            else:
                self.lr_scheduler.step(plateau_metric)
        else:
            self.lr_scheduler.step()

        log['lr_ve'] = self.optimizer.param_groups[0]['lr']
        log['lr_ed'] = self.optimizer.param_groups[1]['lr']

        return log
