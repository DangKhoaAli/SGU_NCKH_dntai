"""
tag_labels.py
Định nghĩa 14 nhãn bệnh lý phổ biến trên IU-Xray và hàm trích xuất
multi-hot label vector từ text báo cáo bằng keyword matching.
"""

import re
import torch

# 14 tags phổ biến nhất trên IU-Xray (theo thứ tự index cố định)
IUXRAY_TAGS = [
    'cardiomegaly',   # 0
    'effusion',       # 1
    'pneumonia',      # 2
    'atelectasis',    # 3
    'edema',          # 4
    'consolidation',  # 5
    'pleural',        # 6
    'opacity',        # 7
    'infiltrate',     # 8
    'pneumothorax',   # 9
    'fracture',       # 10
    'hernia',         # 11
    'nodule',         # 12
    'normal',         # 13  ← "no finding / normal / unremarkable"
]

NUM_TAGS = len(IUXRAY_TAGS)

# Mapping từ tag → danh sách từ khóa đồng nghĩa
_TAG_KEYWORDS = {
    'cardiomegaly':  ['cardiomegaly', 'cardiac enlargement', 'enlarged heart',
                      'cardiomediastinal silhouette is enlarged'],
    'effusion':      ['effusion', 'pleural fluid', 'fluid collection'],
    'pneumonia':     ['pneumonia', 'pneumonic', 'infection', 'lobar consolidation'],
    'atelectasis':   ['atelectasis', 'atelectatic', 'collapse', 'subsegmental'],
    'edema':         ['edema', 'oedema', 'pulmonary edema', 'interstitial edema'],
    'consolidation': ['consolidation', 'airspace disease', 'alveolar'],
    'pleural':       ['pleural', 'pleuritis', 'pleurisy'],
    'opacity':       ['opacity', 'opacification', 'opacities', 'haziness'],
    'infiltrate':    ['infiltrate', 'infiltration', 'infiltrates'],
    'pneumothorax':  ['pneumothorax', 'pneumothoraces'],
    'fracture':      ['fracture', 'rib fracture', 'clavicle fracture'],
    'hernia':        ['hernia', 'hiatal hernia'],
    'nodule':        ['nodule', 'nodular', 'mass', 'lesion'],
    'normal':        ['normal', 'no acute', 'unremarkable', 'no finding',
                      'no evidence of', 'clear', 'within normal limits'],
}

# Biên dịch sang regex một lần duy nhất để tăng tốc
_TAG_PATTERNS = {
    tag: re.compile(r'\b(?:' + '|'.join(re.escape(kw) for kw in kws) + r')\b', re.IGNORECASE)
    for tag, kws in _TAG_KEYWORDS.items()
}


def extract_tags(report_text: str) -> list:
    """
    Trích xuất multi-hot vector từ một chuỗi report.

    Args:
        report_text: Chuỗi báo cáo thô (chưa clean).

    Returns:
        list of int, độ dài NUM_TAGS, mỗi phần tử là 0 hoặc 1.
    """
    labels = []
    for tag in IUXRAY_TAGS:
        match = _TAG_PATTERNS[tag].search(report_text)
        labels.append(1 if match else 0)
    return labels


def extract_tags_batch(reports: list) -> torch.Tensor:
    """
    Trích xuất multi-hot matrix từ danh sách reports.

    Args:
        reports: list of str

    Returns:
        Tensor [len(reports), NUM_TAGS], dtype=float32
    """
    batch = [extract_tags(r) for r in reports]
    return torch.tensor(batch, dtype=torch.float32)
