import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm

from modules.utils import generate_heatmap


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

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

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self._load_model_state(state_dict)


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        model_core = self._get_model()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device, non_blocking=True)
                reports_ids = reports_ids.to(self.device, non_blocking=True)
                reports_masks = reports_masks.to(self.device, non_blocking=True)
                output, _ = self.model(images, mode='sample')
                reports = model_core.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model_core.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

        return log
        
    def translate_by_folder_id(self, folder_ids):
        """
        Dịch các mẫu có ``image_id`` (tên folder chứa ảnh) nằm trong ``folder_ids``.
        Args:
            folder_ids (list[str]): Danh sách tên folder cần dịch.
                Ví dụ: ['CXR1_1_IM-0001', 'CXR3_1_IM-0005']
        Cách dùng:
            tester.translate_by_folder_id(['CXR1_1_IM-0001', 'CXR3_1_IM-0005'])
        """
        self.logger.info('=== Chế độ dịch theo folder ID ===')
        self.model.eval()
        model_core = self._get_model()
        target_set = set(folder_ids)
        found = {}   # folder_id -> (gt, pred)
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                # Lọc các vị trí trong batch khớp với folder_ids cần tìm
                local_mask = [i for i, fid in enumerate(images_id) if fid in target_set]
                if not local_mask:
                    if len(found) >= len(target_set):
                        break
                    continue
                sel_images = images[local_mask].to(self.device, non_blocking=True)
                sel_ids    = reports_ids[local_mask].to(self.device, non_blocking=True)
                output, _ = self.model(sel_images, mode='sample')
                preds = model_core.tokenizer.decode_batch(output.cpu().numpy())
                gts   = model_core.tokenizer.decode_batch(sel_ids[:, 1:].cpu().numpy())
                for li, gi in enumerate(local_mask):
                    fid = images_id[gi]
                    if fid not in found:  # tránh ghi đè nếu trùng batch
                        found[fid] = (gts[li], preds[li])
                if len(found) >= len(target_set):
                    break
        # Cảnh báo nếu không tìm thấy
        not_found = target_set - set(found.keys())
        if not_found:
            self.logger.warning(
                'Không tìm thấy các folder sau trong test set: {}'.format(sorted(not_found))
            )
        # In kết quả
        sep = '=' * 70
        print('\n' + sep)
        print(f'  KẾT QUẢ DỊCH THEO FOLDER ID  —  {len(found)} mẫu')
        print(sep)
        for i, fid in enumerate(folder_ids):
            if fid not in found:
                print(f'\n[Mẫu {i + 1}]  Folder: {fid}')
                print('  *** Không tìm thấy trong test set ***')
                continue
            gt, pred = found[fid]
            print(f'\n[Mẫu {i + 1}]  Folder: {fid}')
            print(f'  GROUND TRUTH : {gt}')
            print(f'  SINH RA      : {pred}')
        print('\n' + sep + '\n')
        return {fid: found[fid] for fid in folder_ids if fid in found}
    def translate_samples(self, n=None, indices=None):
        """
        Dịch một số mẫu từ test set và in kết quả ra màn hình.
        Args:
            n (int, optional): Số mẫu cần dịch (lấy từ đầu dataset).
                               Nếu không truyền thì dùng ``indices``.
            indices (list[int], optional): Danh sách index cụ thể trong test set.
                               Ưu tiên hơn ``n`` nếu cả hai đều được truyền.
        Cách dùng:
            tester.translate_samples(n=5)           # dịch 5 mẫu đầu
            tester.translate_samples(indices=[0, 3, 7])  # dịch các mẫu theo index
        """
        self.logger.info('=== Chế độ dịch mẫu (translate_samples) ===')
        self.model.eval()
        model_core = self._get_model()
        # Tập hợp tất cả batch rồi lọc theo yêu cầu
        all_images_id, all_reports, all_gts = [], [], []
        global_idx = 0
        target_set = set(indices) if indices is not None else None
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                batch_sz = images.size(0)
                batch_indices = list(range(global_idx, global_idx + batch_sz))
                # Xác định các sample nào trong batch này cần lấy
                if target_set is not None:
                    local_mask = [i for i, gi in enumerate(batch_indices) if gi in target_set]
                elif n is not None:
                    local_mask = [i for i, gi in enumerate(batch_indices) if gi < n]
                else:
                    local_mask = list(range(batch_sz))
                if not local_mask:
                    global_idx += batch_sz
                    # Dừng sớm nếu đã thu thập đủ
                    if target_set is None and n is not None and global_idx >= n:
                        break
                    continue
                sel_images   = images[local_mask].to(self.device, non_blocking=True)
                sel_ids      = reports_ids[local_mask].to(self.device, non_blocking=True)
                output, _ = self.model(sel_images, mode='sample')
                reports   = model_core.tokenizer.decode_batch(output.cpu().numpy())
                gts       = model_core.tokenizer.decode_batch(sel_ids[:, 1:].cpu().numpy())
                for li, gi in enumerate(local_mask):
                    all_images_id.append(images_id[gi])
                    all_reports.append(reports[li])
                    all_gts.append(gts[li])
                global_idx += batch_sz
                # Dừng sớm khi đã đủ mẫu
                collected = len(all_reports)
                if target_set is None and n is not None and collected >= n:
                    break
                if target_set is not None and collected >= len(target_set):
                    break
        # In kết quả
        sep = '=' * 70
        print('\n' + sep)
        print(f'  KẾT QUẢ DỊCH MẪU  —  {len(all_reports)} mẫu')
        print(sep)
        for i, (img_id, gt, pred) in enumerate(zip(all_images_id, all_gts, all_reports)):
            print(f'\n[Mẫu {i + 1}]  ID ảnh: {img_id}')
            print(f'  GROUND TRUTH : {gt}')
            print(f'  SINH RA      : {pred}')
        print('\n' + sep + '\n')
        return list(zip(all_images_id, all_gts, all_reports))


    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "attentions_entities"), exist_ok=True)
        ner = spacy.load("en_core_sci_sm")
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        model_core = self._get_model()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device, non_blocking=True)
                reports_ids = reports_ids.to(self.device, non_blocking=True)
                reports_masks = reports_masks.to(self.device, non_blocking=True)
                output, _ = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = model_core.tokenizer.decode_batch(output.cpu().numpy())[0].split()

                char2word = [idx for word_idx, word in enumerate(report) for idx in [word_idx] * (len(word) + 1)][:-1]

                attention_weights = model_core.encoder_decoder.attention_weights[:-1]
                assert len(attention_weights) == len(report)
                for word_idx, (attns, word) in enumerate(zip(attention_weights, report)):
                    for layer_idx, attn in enumerate(attns):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn.mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)

                for ne_idx, ne in enumerate(ner(" ".join(report)).ents):
                    for layer_idx in range(len(attention_weights[0])):
                        os.makedirs(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)
                        attn = [attns[layer_idx] for attns in
                                attention_weights[char2word[ne.start_char]:char2word[ne.end_char] + 1]]
                        attn = np.concatenate(attn, axis=2)
                        heatmap = generate_heatmap(image, attn.mean(1).mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(ne_idx, ne)),
                                    heatmap)
