import math
from typing import Dict, List, Optional


class GraphLiteReportWeighter:
    def __init__(
        self,
        use_tfidf_weight: bool = False,
        idf_dict: Optional[Dict[str, float]] = None,
        tfidf_alpha: float = 0.05,
        tfidf_max_factor: float = 1.10,
    ):
        self.default_weight = 1.0
        self.use_tfidf_weight = bool(use_tfidf_weight)
        self.idf_dict = idf_dict or {}
        self.tfidf_alpha = max(0.0, float(tfidf_alpha))
        self.tfidf_max_factor = max(1.0, float(tfidf_max_factor))
        finite_idfs = [
            float(value)
            for value in self.idf_dict.values()
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
        ]
        self.max_idf = max(finite_idfs) if finite_idfs else 0.0

        self.finding_weights = {
            'pneumothorax': 2.0,
            'effusion': 1.9,
            'edema': 1.9,
            'cardiomegaly': 1.7,
            'consolidation': 1.7,
            'atelectasis': 1.5,
            'opacity': 1.4,
            'infiltrate': 1.5,
            'fracture': 2.0,
            'nodule': 1.7,
            'mass': 1.9,
        }

        self.anatomy_weights = {
            'lung': 1.1,
            'lungs': 1.1,
            'heart': 1.1,
            'cardiac': 1.1,
            'pleural': 1.1,
            'mediastinal': 1.1,
            'left': 1.05,
            'right': 1.05,
            'upper': 1.05,
            'lower': 1.05,
            'base': 1.05,
            'bases': 1.05,
        }

        self.negation_weights = {
            'no': 1.5,
            'without': 1.5,
            'absent': 1.5,
            'negative': 1.4,
        }

        self.uncertainty_weights = {
            'possible': 1.25,
            'possibly': 1.25,
            'may': 1.2,
            'could': 1.2,
            'suggest': 1.2,
            'suggesting': 1.2,
        }

    def get_tfidf_factor(self, token: str) -> float:
        if not self.use_tfidf_weight or self.max_idf <= 0.0:
            return 1.0

        idf = self.idf_dict.get(token)
        if idf is None:
            return 1.0

        idf = float(idf)
        if not math.isfinite(idf) or idf <= 0.0:
            return 1.0

        normalized_idf = min(1.0, max(0.0, idf / self.max_idf))
        factor = 1.0 + self.tfidf_alpha * normalized_idf
        factor = min(self.tfidf_max_factor, max(1.0, factor))
        if not math.isfinite(factor):
            return 1.0
        return factor

    def apply_tfidf_factor(self, tokens: List[str], weights: List[float]) -> List[float]:
        if not self.use_tfidf_weight:
            return weights

        final_weights = []
        for i, weight in enumerate(weights):
            token = tokens[i] if i < len(tokens) else ''
            weight = float(weight)
            if not math.isfinite(weight):
                weight = self.default_weight
            factor = self.get_tfidf_factor(token)
            final_weight = max(self.default_weight, weight * factor)
            if not math.isfinite(final_weight):
                final_weight = self.default_weight
            final_weights.append(final_weight)

        return final_weights

    def build_token_weights_from_tokens(self, tokens: List[str]) -> List[float]:
        weights = [self.default_weight] * len(tokens)

        for i, token in enumerate(tokens):
            if token in self.finding_weights:
                weights[i] = max(weights[i], self.finding_weights[token])
            if token in self.anatomy_weights:
                weights[i] = max(weights[i], self.anatomy_weights[token])
            if token in self.negation_weights:
                weights[i] = max(weights[i], self.negation_weights[token])
            if token in self.uncertainty_weights:
                weights[i] = max(weights[i], self.uncertainty_weights[token])

            if token in self.finding_weights:
                left = max(0, i - 3)
                for j in range(left, i):
                    context_token = tokens[j]
                    if context_token in self.negation_weights:
                        weights[j] = max(weights[j], self.negation_weights[context_token])
                    if context_token in self.uncertainty_weights:
                        weights[j] = max(weights[j], self.uncertainty_weights[context_token])

        return self.apply_tfidf_factor(tokens, weights)

    def build_sequence_weights(self, report: str, tokenizer, max_seq_length: int) -> List[float]:
        ids = tokenizer(report)[:max_seq_length]
        tokens = tokenizer.clean_report(report).split()
        token_weights = self.build_token_weights_from_tokens(tokens)
        sequence_weights = [self.default_weight] + token_weights + [self.default_weight]

        if len(sequence_weights) < len(ids):
            sequence_weights.extend([self.default_weight] * (len(ids) - len(sequence_weights)))

        return sequence_weights[:len(ids)]
