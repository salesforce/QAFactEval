# QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization

This is the official code repository for the NAACL 2022 paper [QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization](https://arxiv.org/abs/2112.08542)
by [Alexander R. Fabbri](https://twitter.com/alexfabbri4), [Chien-Sheng Wu](https://twitter.com/jasonwu0731), [Wenhao Liu](https://twitter.com/owenhaoliu), and [Caiming Xiong](https://twitter.com/caimingxiong). 

In our paper, we conduct an extensive comparison of the components of QA-based metrics for factual consistency evaluation in summarization. Our optimized metric builds on [QAEval](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00397/106792/Towards-Question-Answering-as-an-Automatic-Metric) with question consistency filtering and an improved answer overlap metric, leading to a 14% average improvement over previous QA-based metrics on the [SummaC](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00453/109470/SummaC-Re-Visiting-NLI-based-Models-for) factual consistency benchmark. 

## Table of Contents

1. [Updates](#updates)
2. [Using QAFactEval](#using-qafacteval)
3. [Citation](#citation)
4. [License](#license)


## Updates
_5/2/2022_ - Initial commit! :) 

## Using QAFactEval

You can install qafacteval via pip:
```bash
pip install qafacteval
```

You can also install from source:

```bash
git clone https://github.com/salesforce/QAFactEval
cd QAFactEval
pip install -e .
```

### For use in scripts
Download the required pretrained models using `download_models.sh`.

See `run.py` for an example of using the QAFactEval metric:

```python
from qafacteval import QAFactEval
kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}

model_folder = "" # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)

results = metric.score_batch(["This is a source document"], [["This is a summary."]], return_qa_pairs=True)
score = results[0][0]['qa-eval']['lerc_quip']

```

## Citation

When referencing this repository, please cite [this paper](https://arxiv.org/abs/2112.08542):

```bibtex
@misc{fabbri-etal-2022-qafacteval,
  title = {QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization},
  author = {Alexander R. Fabbri and Chien-Sheng Wu and Wenhao Liu and Caiming Xiong},
  year={2022},
  eprint={2112.08542},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url = {https://arxiv.org/abs/2112.08542},
}
```

## License

This repository is released under the [BSD-3 License](LICENSE.txt).
