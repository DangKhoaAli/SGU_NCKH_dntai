import json
import os
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset

from .report_weighting import GraphLiteReportWeighter


DEFAULT_IUXRAY_VIEW_FILTER_CSV = '/kaggle/input/datasets/quooccuongwf/dataset-errors/iu_xray_select_2views_by_cosine.csv'
LOCAL_IUXRAY_VIEW_FILTER_CSV = os.path.join('logs', 'iu_xray_select_2views_by_cosine.csv')


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_weighted_nll = int(getattr(args, 'use_weighted_nll', 0)) == 1
        self.report_weighter = GraphLiteReportWeighter() if self.use_weighted_nll else None
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            if self.use_weighted_nll:
                self.examples[i]['report_weights'] = self.report_weighter.build_sequence_weights(
                    self.examples[i]['report'], tokenizer, self.max_seq_length
                )
                assert len(self.examples[i]['report_weights']) == len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super().__init__(args, tokenizer, split, transform)
        self.use_view_filter = self._as_bool(getattr(args, 'use_iu_xray_view_filter', True))
        self.view_filter_csv = getattr(
            args,
            'iu_xray_view_filter_csv',
            DEFAULT_IUXRAY_VIEW_FILTER_CSV
        )
        self.view_filter_csv = self._resolve_view_filter_csv(self.view_filter_csv)
        self._apply_view_filter()

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

    def _resolve_view_filter_csv(self, view_filter_csv):
        if not view_filter_csv or os.path.exists(view_filter_csv):
            return view_filter_csv

        if view_filter_csv == DEFAULT_IUXRAY_VIEW_FILTER_CSV and os.path.exists(LOCAL_IUXRAY_VIEW_FILTER_CSV):
            return LOCAL_IUXRAY_VIEW_FILTER_CSV

        ann_dir = os.path.dirname(os.path.abspath(self.ann_path))
        repo_root = os.path.dirname(os.path.dirname(ann_dir))
        candidate = os.path.join(repo_root, view_filter_csv)
        if os.path.exists(candidate):
            return candidate

        return view_filter_csv

    def _apply_view_filter(self):
        if not self.use_view_filter:
            print(f"[IUXRAY view filter][{self.split}] disabled")
            return

        if not self.view_filter_csv or not os.path.exists(self.view_filter_csv):
            print(f"[IUXRAY view filter][{self.split}] no filter csv found: {self.view_filter_csv}")
            return

        keep_by_folder = {}
        drop_by_folder = {}
        with open(self.view_filter_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder = row.get('folder', '')
                keep = [name for name in row.get('keep', '').split(';') if name]
                drop = [name for name in row.get('drop', '').split(';') if name]
                if folder and len(keep) == 2:
                    keep_by_folder[folder] = keep
                    drop_by_folder[folder] = drop

        filtered_studies = 0
        removed_original_paths = 0
        missing_folders = 0

        for example in self.examples:
            folder = example['id']
            if folder not in keep_by_folder:
                continue

            keep_paths = [os.path.join(folder, image_name) for image_name in keep_by_folder[folder]]
            missing = [
                image_path for image_path in keep_paths
                if not os.path.exists(os.path.join(self.image_dir, image_path))
            ]
            if missing:
                missing_folders += 1
                continue

            old_paths = {os.path.normpath(path) for path in example.get('image_path', [])}
            example['image_path'] = keep_paths
            filtered_studies += 1
            removed_original_paths += len(old_paths - {os.path.normpath(path) for path in keep_paths})

        manifest_drop_count = sum(len(drop_by_folder.get(example['id'], [])) for example in self.examples)
        print(
            f"[IUXRAY view filter][{self.split}] filtered studies={filtered_studies}, "
            f"drop images by manifest={manifest_drop_count}, "
            f"removed original annotation paths={removed_original_paths}, "
            f"csv={self.view_filter_csv}"
        )
        if missing_folders:
            print(f"[IUXRAY view filter][{self.split}] skipped {missing_folders} studies because selected files were missing")

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        if self.use_weighted_nll:
            report_weights = example['report_weights']
            return image_id, image, report_ids, report_masks, report_weights, seq_length
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        if self.use_weighted_nll:
            report_weights = example['report_weights']
            return image_id, image, report_ids, report_masks, report_weights, seq_length
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
