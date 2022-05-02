"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from typing import Dict, List, Set

from qaeval.scoring.scorers.scorer import Scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict(our_model, our_tokenizer, batch_sentences, device):
    inputs = our_tokenizer(batch_sentences, max_length=512,  truncation=True, \
        padding="max_length", return_tensors="pt")
    outputs = our_model(input_ids=inputs["input_ids"].to(device), \
        attention_mask=inputs["attention_mask"].to(device))
    outputs = [x[0] for x in outputs[0].cpu().tolist()]
    outputs = [{"pred_score": x} for x in outputs]

    return outputs


class LERCQuipScorer(Scorer):
    def __init__(self, lerc_quip_path: str, cuda_device: int, batch_size: int = 8) -> None:
        self.device = cuda_device

        self.predictor = AutoModelForSequenceClassification.from_pretrained(lerc_quip_path).to(self.device)
        self.predictor.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(lerc_quip_path)
        self.batch_size = batch_size

    def keys(self) -> Set[str]:
        return {'lerc_quip'}

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        input_dicts = []
        indices = []
        for i, (answer, question, prediction, probability, null_probability) in \
                enumerate(zip(answers, questions, predictions,
                    probabilities, null_probabilities)):
            if probability > null_probability:
                sentence1 = f"{question} <q> {answer} <r> {prediction} <c> {context}"
                input_dicts.append(sentence1)
                indices.append(i)

        output_dicts = []
        for i in range(0, len(input_dicts), self.batch_size):
            batch = input_dicts[i:i + self.batch_size]
            output_dicts.extend(predict(self.predictor, self.tokenizer, batch, self.device))
        assert len(output_dicts) == len(input_dicts)

        scores = [0.0] * len(questions)
        for i, output_dict in zip(indices, output_dicts):
            scores[i] = output_dict['pred_score']
        scores = [{'lerc_quip': s} for s in scores]
        return scores
