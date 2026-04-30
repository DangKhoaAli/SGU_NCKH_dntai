import json
import math
import re
from collections import Counter


class Tokenizer(object):
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    # thêm các token đặc biệt thay vì luôn gán id =0
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.vocab_method = getattr(args, 'vocab_method', 'threshold')
        self.tfidf_top_k = getattr(args, 'tfidf_top_k', 0)
        self.tfidf_min_score = getattr(args, 'tfidf_min_score', 0.0)
        self.tfidf_min_df = getattr(args, 'tfidf_min_df', 1)
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'mimic_cxr':
            self.clean_report = self.clean_report_mimic_cxr
        else:
            raise ValueError(f'Unsupported dataset_name: {self.dataset_name}')

        with open(self.ann_path, 'r', encoding='utf-8') as f:
            self.ann = json.load(f)

        self.token2idx, self.idx2token = self.create_vocabulary()
        self.pad_idx = self.token2idx[self.PAD_TOKEN]
        self.bos_idx = self.token2idx[self.BOS_TOKEN]
        self.eos_idx = self.token2idx[self.EOS_TOKEN]
        self.unk_idx = self.token2idx[self.UNK_TOKEN]

        # Keep model sampling args consistent with the tokenizer vocabulary.
        args.pad_idx = self.pad_idx
        args.bos_idx = self.bos_idx
        args.eos_idx = self.eos_idx

    def create_vocabulary(self):
        train_documents = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            train_documents.append(tokens)

        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        if self.vocab_method == 'threshold':
            vocab = self.create_threshold_vocabulary(train_documents, special_tokens)
        elif self.vocab_method == 'tfidf':
            vocab = self.create_tfidf_vocabulary(train_documents, special_tokens)
        else:
            raise ValueError(f'Unsupported vocab_method: {self.vocab_method}')

        vocab.sort()
        vocab = special_tokens + vocab
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token

    def create_threshold_vocabulary(self, train_documents, special_tokens):
        total_tokens = []
        for tokens in train_documents:
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        return [token for token, freq in counter.items() if freq >= self.threshold and token not in special_tokens]

    def create_tfidf_vocabulary(self, train_documents, special_tokens):
        num_documents = len(train_documents)
        doc_freq = Counter()
        term_freqs = []

        for tokens in train_documents:
            term_freq = Counter(tokens)
            term_freqs.append(term_freq)
            doc_freq.update(term_freq.keys())

        tfidf_scores = Counter()
        for term_freq in term_freqs:
            doc_len = sum(term_freq.values())
            if doc_len == 0:
                continue
            for token, count in term_freq.items():
                if token in special_tokens or doc_freq[token] < self.tfidf_min_df:
                    continue
                tf = count / doc_len
                idf = math.log((1 + num_documents) / (1 + doc_freq[token])) + 1
                tfidf_scores[token] = max(tfidf_scores[token], tf * idf)

        vocab = [
            token
            for token, score in tfidf_scores.items()
            if score >= self.tfidf_min_score
        ]
        if self.tfidf_top_k and self.tfidf_top_k > 0:
            vocab = [
                token
                for token, _ in sorted(
                    ((token, tfidf_scores[token]) for token in vocab),
                    key=lambda item: (-item[1], item[0])
                )[:self.tfidf_top_k]
            ]
        return vocab

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.unk_idx
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.bos_idx):
                continue
            tokens.append(self.idx2token.get(idx, self.UNK_TOKEN))
        return ' '.join(tokens)

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out