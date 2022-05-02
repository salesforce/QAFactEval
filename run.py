"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import argparse
from qafacteval import QAFactEval


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running QAFactEval')
    parser.add_argument('--fname', default=None, required=True)
    parser.add_argument('--outfname', default=None, required=True)
    parser.add_argument('--model_folder', default=None, required=True)
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--use_lerc_quip', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--generation_batch_size', default=32, type=int)
    parser.add_argument('--answering_batch_size', default=32, type=int)
    parser.add_argument('--lerc_batch_size', default=8, type=int)

    args = parser.parse_args()
    kwargs = {"cuda_device": args.cuda_device, "use_lerc_quip": args.use_lerc_quip, \
        "verbose": args.verbose, "generation_batch_size": args.generation_batch_size, \
        "answering_batch_size": args.answering_batch_size, "lerc_batch_size": args.lerc_batch_size}

    metric = QAFactEval(
        lerc_quip_path=f"{args.model_folder}/quip-512-mocha",
        generation_model_path=f"{args.model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{args.model_folder}/answering",
        lerc_model_path=f"{args.model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{args.model_folder}/lerc/pretraining.tar.gz",
        **kwargs
    )

    # results = metric.score_batch(["This is a source document"], \
    #   [["This is a summary."]], return_qa_pairs=True)
    # print(results[0][0]['qa-eval']['lerc_quip'])

    candidates = []
    references_list = []
    datas = []
    with open(args.fname) as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
            doc = data['document']['text']
            summ = data['claim']
            candidates.append(doc)
            references_list.append([summ])

    results = metric.score_batch_qafacteval(candidates, references_list, return_qa_pairs=True)
    with open(args.outfname, "w") as out:
        for (metrics, qa_pairs, qa_pairs_original, predictions_cons), data in zip(results, datas):
            data["metrics"] = metrics
            data["qa_pairs"] = qa_pairs
            data["qa_pairs_nonfiltered"] = qa_pairs_original
            data["qa_summary"] = predictions_cons
            out.write(json.dumps(data) + "\n")
