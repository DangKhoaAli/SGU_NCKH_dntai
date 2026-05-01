from typing import List


class GraphLiteReportWeighter:
    def __init__(self):
        self.default_weight = 1.0

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

        return weights

    def build_sequence_weights(self, report: str, tokenizer, max_seq_length: int) -> List[float]:
        ids = tokenizer(report)[:max_seq_length]
        tokens = tokenizer.clean_report(report).split()
        token_weights = self.build_token_weights_from_tokens(tokens)
        sequence_weights = [self.default_weight] + token_weights + [self.default_weight]

        if len(sequence_weights) < len(ids):
            sequence_weights.extend([self.default_weight] * (len(ids) - len(sequence_weights)))

        return sequence_weights[:len(ids)]
