# Joint Constrained Learning for Event-Event Relation Extraction

This is the repository for the resources in EMNLP 2020 Paper "Joint Constrained Learning for Event-Event Relation Extraction" (https://www.aclweb.org/anthology/2020.emnlp-main.51/). This repository contains the source code and datasets used in our paper.

## Abstract

Understanding natural language involves recognizing how multiple event mentions structurally and temporally interact with each other. In this process, one can induce event complexes that organize multi-granular events with temporal order and membership relations interweaving among them. Due to the lack of jointly labeled data for these relational phenomena and the restriction on the structures they articulate, we propose a joint constrained learning framework for modeling event-event relations. Specifically, the framework enforces logical constraints within and across multiple temporal and subevent relations by converting these constraints into differentiable learning objectives. We show that our joint constrained learning approach effectively compensates for the lack of jointly labeled data, and outperforms SOTA methods on benchmarks for both temporal relation extraction and event hierarchy construction, replacing a commonly used but more expensive global inference process. We also present a promising case study showing the effectiveness of our approach in inducing event complexes on an external corpus.

<p align="center">
    <img src="https://github.com/why2011btv/JointConstrainedLearning/blob/master/Example.jpg?raw=true" alt="drawing" width="500"/>
</p>

## How to run the code
### Environment Setup et al.
```
git clone https://github.com/why2011btv/JointConstrainedLearning.git
conda env create -n conda-env -f environment.yml
pip install -r requirements.txt
python spacy -m en-core-web-sm

mkdir rst_file
mkdir model_params
cd model_params
mkdir HiEve_best
mkdir MATRES_best
cd ..
```
### Running experiments in the paper
`python3 main.py <DEVICE_ID> <BATCH_SIZE> <RESULT_FILE> <EPOCH> <SETTING> <LOSS> <FINETUNE> <MAX_EVALS> <DEBUGGING>`

`<DEVICE_ID>`: choose from "gpu_0", "gpu_1", "gpu_5,6,7", etc.

`<BATCH_SIZE>`: choose from "batch_16" (with finetuning), "batch_500" (w/o finetuning)

`<RESULT_FILE>`: for example, "0920_0.rst"

`<EPOCH>`: choose from "epoch_40", "epoch_80", etc.

`<SETTING>`: choose from "MATRES", "HiEve", "Joint"

`<LOSS>`: choose from "add_loss_0" (w/o constraints), "add_loss_1" (within-task constraints), "add_loss_2" (within & cross task constraints)

`<FINETUNE>`: choose from "finetune_0" (roberta-base emb w/o finetuning + BiLSTM), "finetune_1" (roberta-base emb with finetuning, no BiLSTM)

`<MAX_EVALS>`: number of times for hyperopt to run experiments for finding best hyperparameters, choose from "MAX_EVALS_50", etc.

`<DEBUGGING>`: whether to debug, choose from "debugging_0", "debugging_1"

### Example commands
#### Command for "Joint Constrained Learning" with all constraints and RoBERTa finetuning
`nohup python3 main.py gpu_0 batch_16 0118_0.rst epoch_40 Joint add_loss_2 finetune_1 MAX_EVALS_50 debugging_0 > output_redirect/0118_0.out 2>&1 &`

To look at the standard output: `cat output_redirect/0118_0.out`


## Reference
Bibtex:
```
@inproceedings{WCZR20,
    author = {Haoyu Wang and Muhao Chen and Hongming Zhang and Dan Roth},
    title = {{Joint Constrained Learning for Event-Event Relation Extraction}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2020},
    url = "https://cogcomp.seas.upenn.edu/papers/WCZR20.pdf",
    funding = {KAIROS},
}
```
