# Joint Constrained Learning for Event-Event Relation Extraction

This is the repository for the resources in EMNLP 2020 Paper ["Joint Constrained Learning for Event-Event Relation Extraction"](https://www.aclweb.org/anthology/2020.emnlp-main.51/). This repository contains the source code and datasets used in our paper.

## Abstract

Understanding natural language involves recognizing how multiple event mentions structurally and temporally interact with each other. In this process, one can induce event complexes that organize multi-granular events with temporal order and membership relations interweaving among them. Due to the lack of jointly labeled data for these relational phenomena and the restriction on the structures they articulate, we propose a joint constrained learning framework for modeling event-event relations. Specifically, the framework enforces logical constraints within and across multiple temporal and subevent relations by converting these constraints into differentiable learning objectives. We show that our joint constrained learning approach effectively compensates for the lack of jointly labeled data, and outperforms SOTA methods on benchmarks for both temporal relation extraction and event hierarchy construction, replacing a commonly used but more expensive global inference process. We also present a promising case study showing the effectiveness of our approach in inducing event complexes on an external corpus.

<p align="center">
    <img src="https://github.com/why2011btv/JointConstrainedLearning/blob/master/example/Example.jpg?raw=true" alt="drawing" width="500"/>
</p>

## Dataset

Two datasets ([MATRES](https://github.com/why2011btv/JointConstrainedLearning/tree/master/MATRES) and [HiEve](https://github.com/why2011btv/JointConstrainedLearning/tree/master/hievents_v2)) are used for training in the paper. 

## How to train
### Environment Setup et al.
```
git clone https://github.com/why2011btv/JointConstrainedLearning.git
conda env create -n conda-env -f env/environment.yml
pip install -r env/requirements.txt
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
#### Command for "Joint Constrained Learning" with all constraints and RoBERTa finetuning, using hyperopt for finding best hyperparameters
`nohup python3 main.py gpu_0 batch_16 0118_0.rst epoch_40 Joint add_loss_2 finetune_1 MAX_EVALS_50 debugging_0 > output_redirect/0118_0.out 2>&1 &`

To look at the standard output: `cat output_redirect/0118_0.out`

#### Command for training using a specific set of hyperparameters
`nohup python3 main.py gpu_0 batch_16 0118_1.rst epoch_40 Joint add_loss_2 finetune_1 MAX_EVALS_1 debugging_1 > output_redirect/0118_1.out 2>&1 &`

You need to specify the hyperparameters in [main.py](https://github.com/why2011btv/JointConstrainedLearning/blob/56818c48e50af01a6b2f85252a91cf9e2c20fbf7/main.py#L51). You can find the hyperparameter settings under [config](https://github.com/why2011btv/JointConstrainedLearning/tree/master/config) folder.

## How to predict

### Input & Output

Input should be a json file that contains a list of dictionaries. Each dictionary contains four key-value pairs, i.e., two sentences and two char id's denoting the start position of events. Two examples can be found under [example](https://github.com/why2011btv/JointConstrainedLearning/tree/master/example) folder, you can also find more details on generating input json file in [example/example_input_for_predict.ipynb](https://github.com/why2011btv/JointConstrainedLearning/blob/master/example/example_input_for_predict.ipynb).

Output will also be a json file under [output](https://github.com/why2011btv/JointConstrainedLearning/tree/master/output) folder. The output contains a dictionary with two key-value pairs; one is labels, the other is predicted probabilities.

### How to run 
`python3 predict.py <INPUT_FILE> <TASK> <MODEL> <OUTPUT_FILE>`

`<INPUT_FILE>`: a json file

`<TASK>`: choose from "subevent" and "temporal"

`<MODEL>`: choose from "MATRES", "HiEve", "Joint" (i.e., dataset on which the model was trained)

`<OUTPUT_FILE>`: name for a json file

### Example commands
#### Command for predicting temporal relations
`python predict.py example/temporal_example_input.json temporal MATRES predict_temporal.json`
#### Command for predicting subevent relations
`python predict.py example/subevent_example_input.json subevent Joint predict_subevent.json`

#### [Link to pre-trained model](https://drive.google.com/drive/folders/1PyNAlNHY144pGsko9iYxwYlqf4ud0Lq1?usp=sharing)


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
