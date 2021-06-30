import tqdm
import time
import datetime
import random
import numpy as np
from document_reader import *
from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
import sys
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model import *
from metric import metric, CM_metric
from exp import *
from data import *
from util import *
import json

torch.manual_seed(42)

### Read parameters ###
if len(sys.argv) > 1:
    gpu_num, batch_size, rst_file_name, epochs, dataset, add_loss, finetune, MAX_EVALS, debugging = sys.argv[1][4:], int(sys.argv[2][6:]), sys.argv[3], int(sys.argv[4][6:]), sys.argv[5], int(sys.argv[6][9:]), int(sys.argv[7][9:]), int(sys.argv[8][10:]), int(sys.argv[9][10:])
    
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
cuda = torch.device('cuda')
writer = SummaryWriter(comment=rst_file_name.replace(".rst", ""))

### restore model ###
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
MATRES_best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt") 
#load_model_path = model_params_dir + dataset + "_best/" # for manual test

# Use hyperopt to select the best hyperparameters
import hyperopt
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import csv
from timeit import default_timer as timer

if debugging:
    space = {
        'downsample': 0.01,
        'learning_rate': 0.0000001,
        'lambda_annoT': 2.0,
        'lambda_annoH': 2.0,
        'lambda_transT': 1.0,
        'lambda_transH': 1.0,
        'lambda_cross': 1.0,
        'MLP_size': 512, #hp.quniform('MLP_size', 128, 1024, 1),
        'num_layers': 1, #hp.quniform('num_layers', 1, 4, 1),
        'lstm_hidden_size': 256, #hp.quniform('lstm_hidden_size', 128, 512, 1),
        'roberta_hidden_size': 1024, #hp.quniform('roberta_hidden_size', 768, 1024, 1),
        'lstm_input_size': 768, #hp.quniform('lstm_input_size', 768, 1024, 1), # pre-trained word embeddings, roberta-base
    }
    with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
        json.dump(space, config_file)
else:
    space = {
        'downsample': hp.uniform('downsample', 0.01, 0.02),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00000005), np.log(0.0000002)),
        'lambda_annoT': hp.uniform('lambda_annoT', 0.0, 1.0),
        'lambda_annoH': hp.uniform('lambda_annoH', 0.0, 1.0),
        'lambda_transT': hp.uniform('lambda_transT', 0.0, 1.0),
        'lambda_transH': hp.uniform('lambda_transH', 0.0, 1.0),
        'lambda_cross': hp.uniform('lambda_cross', 0.0, 1.0),
        'MLP_size': 512, #hp.quniform('MLP_size', 128, 1024, 1),
        'num_layers': 1, #hp.quniform('num_layers', 1, 4, 1),
        'lstm_hidden_size': 256, #hp.quniform('lstm_hidden_size', 128, 512, 1),
        'roberta_hidden_size': 1024, #hp.quniform('roberta_hidden_size', 768, 1024, 1),
        'lstm_input_size': 768, #hp.quniform('lstm_input_size', 768, 1024, 1), # pre-trained word embeddings, roberta-base
    }
    #with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
    #    json.dump(space, config_file)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def objective(params):
    """Objective function for Hyperparameter Optimization"""
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    start = timer()
    train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, num_classes = data(dataset, debugging, params['downsample'], batch_size)
    if finetune:
        model = roberta_mlp(num_classes, dataset, add_loss, params)
    else:
        model = BiLSTM_MLP(num_classes, dataset, add_loss, params)
    model.to(cuda)
    model.zero_grad()
    if len(gpu_num) > 1:
        model = nn.DataParallel(model) # you may try to run the experiments with multiple GPUs
    print("# of parameters:", count_parameters(model))
    model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters
    total_steps = len(train_dataloader) * epochs
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    if dataset == "MATRES":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        matres_exp = exp(cuda, model, epochs, params['learning_rate'], train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, None, None, finetune, dataset, MATRES_best_PATH, None, None, model_name)
        T_F1, H_F1 = matres_exp.train()
        matres_exp.evaluate(eval_data = "MATRES", test = True)
    elif dataset == "HiEve":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        hieve_exp = exp(cuda, model, epochs, params['learning_rate'], train_dataloader, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, None, HiEve_best_PATH, None, model_name)
        T_F1, H_F1 = hieve_exp.train()
        hieve_exp.evaluate(eval_data = "HiEve", test = True)
    elif dataset == "Joint":
        total_steps = len(train_dataloader) * epochs
        print("Total steps: [number of batches] x [number of epochs] =", total_steps)
        joint_exp = exp(cuda, model, epochs, params['learning_rate'], train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, MATRES_best_PATH, HiEve_best_PATH, None, model_name)
        T_F1, H_F1 = joint_exp.train()
        joint_exp.evaluate(eval_data = "HiEve", test = True)
        joint_exp.evaluate(eval_data = "MATRES", test = True)
    else:
        raise ValueError("Currently not supporting this dataset! -_-'")
    
    print(f'Iteration {ITERATION} result: MATRES F1: {T_F1}; HiEve F1: {H_F1}')
    loss = 2 - T_F1 - H_F1
    
    run_time = format_time(timer() - start)
    
    # Write to the csv file ('a' means append)
    print("########################## Append a row to out_file ##########################")
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, T_F1, H_F1, params, epochs, dataset, finetune, batch_size, ITERATION, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'MATRES F1': T_F1, 'HiEve F1': H_F1, \
            'params': params, 'iteration': ITERATION, \
            'train_time': run_time, 'status': STATUS_OK}

# optimization algorithm
tpe_algorithm = tpe.suggest    

# Keep track of results
bayes_trials = Trials()

# File to save first results
out_file = 'hyperopt_results/' + rst_file_name.replace(".rst", "") + '.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'MATRES F1', 'HiEve F1', 'params', 'epochs', 'dataset', 'finetune', 'batch_size', 'iteration', 'train_time'])
of_connection.close()

# Global variable
global  ITERATION
ITERATION = 0

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

results = pd.read_csv(out_file)

# Sort values with best on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
print(results.head())
