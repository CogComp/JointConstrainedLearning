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
from exp_nowrite import *
from data import *
import json

### Read command line parameters ###
if len(sys.argv) > 1:
    input_file, task, dataset, f_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
f_out = "output/" + f_out
if dataset == "Joint":
    rst_file_name = "0322_1.rst" # Suggested for subevent
elif dataset == "HiEve":
    rst_file_name = "0104_5.rst" # Not suggested for subevent
elif dataset == "MATRES":
    rst_file_name = "0104_3.rst" # Suggested for temporal

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cuda = torch.device('cuda')
epochs = 1
params = {'learning_rate': 0.0000001, 'downsample': 0.01, 'roberta_hidden_size': 1024, 'MLP_size': 512}
debugging = 0
batch_size = 64
add_loss = 1
finetune = 1
model_params_dir = "./model_params/"
HiEve_best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt")
MATRES_best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt")
model_name = rst_file_name.replace(".rst", "")
num_classes = 4

print("Processing input data...")
with open(input_file) as f:
    input_list = json.load(f)
test_set = []
for an_instance in input_list:
    x_sent, x_position = subword_id_getter_space_split(an_instance["sent_1"], an_instance["e1_start_char"])
    y_sent, y_position = subword_id_getter_space_split(an_instance["sent_2"], an_instance["e2_start_char"])
    x_sent = padding(x_sent)
    y_sent = padding(y_sent)
    to_append = 0, 0, 0, \
                x_sent, y_sent, x_sent, \
                x_position, y_position, x_position, \
                0, 0, 0, \
                0, 0, 0, 1
    test_set.append(to_append)
test_dataloader = DataLoader(EventDataset(test_set), batch_size=batch_size, shuffle = False)

if task == "temporal":
    print("loading model from " + MATRES_best_PATH + "...")
else:
    print("loading model from " + HiEve_best_PATH + "...")
model = roberta_mlp(num_classes, dataset, add_loss, params)
model.to(cuda)

if task == "temporal":
    exp = exp_nowrite(cuda, model, epochs, params['learning_rate'], None, None, test_dataloader, None, None, finetune, dataset, MATRES_best_PATH, None, None, model_name)
    exp.evaluate(eval_data = "MATRES", test = True, predict = f_out)
elif task == "subevent":
    exp = exp_nowrite(cuda, model, epochs, params['learning_rate'], None, None, None, None, test_dataloader, finetune, dataset, None, HiEve_best_PATH, None, model_name)
    exp.evaluate(eval_data = "HiEve", test = True, predict = f_out)
else:
    # HiEve test set
    train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, num_classes = data("HiEve", debugging, params['downsample'], batch_size)
    exp = exp_nowrite(cuda, model, epochs, params['learning_rate'], train_dataloader, None, None, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, "Joint", None, HiEve_best_PATH, None, model_name)
    exp.evaluate(eval_data = "HiEve", test = True)
