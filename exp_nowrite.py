import tqdm
import time
import datetime
import random
import numpy as np
from document_reader import *
import os
from os import listdir
from os.path import isfile, join
from EventDataset import EventDataset
import sys
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from metric import metric, CM_metric
from transformers import RobertaModel
import os.path
from os import path
import json
from json import JSONEncoder
import notify
from notify_message import *
from notify_smtp import *
from util import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class exp_nowrite:
    def __init__(self, cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader_MATRES, test_dataloader_MATRES, valid_dataloader_HIEVE, test_dataloader_HIEVE, finetune, dataset, MATRES_best_PATH, HiEve_best_PATH, load_model_path, model_name = None, roberta_size = "roberta-base"):
        self.cuda = cuda
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.finetune = finetune
        self.train_dataloader = train_dataloader
        self.valid_dataloader_MATRES = valid_dataloader_MATRES
        self.test_dataloader_MATRES = test_dataloader_MATRES
        self.valid_dataloader_HIEVE = valid_dataloader_HIEVE
        self.test_dataloader_HIEVE = test_dataloader_HIEVE
        ### fine-tune roberta or not ###
        # if finetune is False, we use fixed roberta embeddings before bilstm and mlp
        self.roberta_size = roberta_size
        if not self.finetune:
            self.RoBERTaModel = RobertaModel.from_pretrained(self.roberta_size).to(self.cuda)
        if self.roberta_size == 'roberta-base':
            self.roberta_dim = 768
        else:
            self.roberta_dim = 1024
            
        self.MATRES_best_micro_F1 = -0.000001
        self.MATRES_best_cm = []
        self.MATRES_best_PATH = MATRES_best_PATH
        
        self.HiEve_best_F1 = -0.000001
        self.HiEve_best_prfs = []
        self.HiEve_best_PATH = HiEve_best_PATH
        
        self.load_model_path = load_model_path
        self.model_name = model_name
        self.best_epoch = 0
        #self.file = open("./rst_file/" + model_name + ".rst", "w")
        #import pdb;pdb.set_trace()
        print("calling exp_nowrite...")
        
    def my_func(self, x_sent):
        my_list = []
        for sent in x_sent:
            my_list.append(self.RoBERTaModel(sent.unsqueeze(0))[0].view(-1, self.roberta_dim))
        return torch.stack(my_list).to(self.cuda)
    
    def train(self):
        total_t0 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True) # AMSGrad
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.model.train()
            self.total_train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                x_sent = batch[3].to(self.cuda)
                #print(x_sent)
                y_sent = batch[4].to(self.cuda)
                z_sent = batch[5].to(self.cuda)
                x_position = batch[6].to(self.cuda)
                y_position = batch[7].to(self.cuda)
                z_position = batch[8].to(self.cuda)
                xy = batch[12].to(self.cuda)
                yz = batch[13].to(self.cuda)
                xz = batch[14].to(self.cuda)
                flag = batch[15].to(self.cuda)
                if self.finetune:
                    alpha_logits, beta_logits, gamma_logits, loss = self.model(x_sent, y_sent, z_sent, x_position, y_position, z_position, xy, yz, xz, flag, loss_out = True)
                else:
                    with torch.no_grad():
                        x_sent_e = self.my_func(x_sent)
                        y_sent_e = self.my_func(y_sent)
                        z_sent_e = self.my_func(z_sent)
                    alpha_logits, beta_logits, gamma_logits, loss = self.model(x_sent_e, y_sent_e, z_sent_e, x_position, y_position, z_position, xy = xy, yz = yz, xz = xz, flag = flag, loss_out = True)
                self.total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            if self.dataset in ["HiEve", "MATRES"]:
                flag = self.evaluate(self.dataset)
            else:
                flag = self.evaluate("HiEve")
                flag = self.evaluate("MATRES")
            if flag == 1:
                self.best_epoch = epoch_i
        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        if self.dataset in ["MATRES", "Joint"]:
            print("  MATRES best micro F1: {0:.3f}".format(self.MATRES_best_micro_F1))
            print("  MATRES best confusion matrix:\n", self.MATRES_best_cm)
            #print("  Dev best:", file = self.file)
            #print("  MATRES best micro F1: {0:.3f}".format(self.MATRES_best_micro_F1), file = self.file)
            #print("  MATRES best confusion matrix:", file = self.file)
            #print(self.MATRES_best_cm, file = self.file)
        if self.dataset in ["HiEve", "Joint"]:
            print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1))
            print("  HiEve best precision_recall_fscore_support:\n", self.HiEve_best_prfs)
            #print("  Dev best:", file = self.file)
            #print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1), file = self.file)
            #print("  HiEve best precision_recall_fscore_support:", file = self.file)
            #print(self.HiEve_best_prfs, file = self.file)
        return self.MATRES_best_micro_F1, self.HiEve_best_F1
            
    def evaluate(self, eval_data, test = False, predict = False):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.
        t0 = time.time()
            
        if test:
            if self.load_model_path:
                self.model = torch.load(self.load_model_path + self.model_name + ".pt")
            elif eval_data == "HiEve":
                self.model = torch.load(self.HiEve_best_PATH)
            else: # MATRES
                self.model = torch.load(self.MATRES_best_PATH)
            self.model.to(self.cuda)
            print("")
            print("loaded " + eval_data + " best model:" + self.model_name + ".pt")
            if predict == False:
                print("(from epoch " + str(self.best_epoch) + " )")
            print("Running Evaluation on " + eval_data + " Test Set...")
            if eval_data == "MATRES":
                dataloader = self.test_dataloader_MATRES
            else:
                dataloader = self.test_dataloader_HIEVE
        else:
            # Evaluation
            print("")
            print("Running Evaluation on Validation Set...")
            if eval_data == "MATRES":
                dataloader = self.valid_dataloader_MATRES
            else:
                dataloader = self.valid_dataloader_HIEVE
            
        self.model.eval()
        
        y_pred = []
        y_gold = []
        y_logits = np.array([[0.0, 1.0, 2.0, 3.0]])
        softmax = nn.Softmax(dim=1)
        # Evaluate data for one epoch
        for batch in dataloader:
            x_sent = batch[3].to(self.cuda)
            y_sent = batch[4].to(self.cuda)
            z_sent = batch[5].to(self.cuda)
            x_position = batch[6].to(self.cuda)
            y_position = batch[7].to(self.cuda)
            z_position = batch[8].to(self.cuda)
            xy = batch[12].to(self.cuda)
            yz = batch[13].to(self.cuda)
            xz = batch[14].to(self.cuda)
            flag = batch[15].to(self.cuda)
            with torch.no_grad():
                if self.finetune:
                    alpha_logits, beta_logits, gamma_logits = self.model(x_sent, y_sent, z_sent, x_position, y_position, z_position, xy, yz, xz, flag, loss_out = None)
                else:
                    with torch.no_grad():
                        x_sent_e = self.my_func(x_sent)
                        y_sent_e = self.my_func(y_sent)
                        z_sent_e = self.my_func(z_sent)
                    alpha_logits, beta_logits, gamma_logits = self.model(x_sent_e, y_sent_e, z_sent_e, x_position, y_position, z_position, xy = xy, yz = yz, xz = xz, flag = flag, loss_out = None)
                    
            if self.dataset == "Joint":
                assert list(alpha_logits.size())[1] == 8
                if eval_data == "MATRES":
                    alpha_logits = torch.narrow(alpha_logits, 1, 4, 4)
                else:
                    alpha_logits = torch.narrow(alpha_logits, 1, 0, 4)
            else:
                assert list(alpha_logits.size())[1] == 4
            # Move logits and labels to CPU
            label_ids = xy.to('cpu').numpy()
            y_predict = torch.max(alpha_logits, 1).indices.cpu().numpy()
            y_pred.extend(y_predict)
            y_gold.extend(label_ids)
            y_logits = np.append(y_logits, softmax(alpha_logits).cpu().numpy(), 0)
                
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))
        
        if predict:
            with open(predict, 'w') as outfile:
                if eval_data == "MATRES":
                    numpyData = {"labels": "0 -- Before; 1 -- After; 2 -- Equal; 3 -- Vague", "array": y_logits}
                else:
                    numpyData = {"labels": "0 -- Parent-Child; 1 -- Child-Parent; 2 -- Coref; 3 -- NoRel", "array": y_logits}
                json.dump(numpyData, outfile, cls=NumpyArrayEncoder)
            msg = message(subject=eval_data + " Prediction Notice",
                          text=self.dataset + "/" + self.model_name + " Predicted " + str(y_logits.shape[0] - 1) + " instances. (Current Path: " + os.getcwd() + ")")
            send(msg)  # and send it
            return 0
        
        if eval_data == "MATRES":
            Acc, P, R, F1, CM = metric(y_gold, y_pred)
            print("  P: {0:.3f}".format(P))
            print("  R: {0:.3f}".format(R))
            print("  F1: {0:.3f}".format(F1))
            if test:
                #print("Test result:", file = self.file)
                #print("  P: {0:.3f}".format(P), file = self.file)
                #print("  R: {0:.3f}".format(R), file = self.file)
                #print("  F1: {0:.3f}".format(F1), file = self.file)
                #print("  Confusion Matrix", file = self.file)
                #print(CM, file = self.file)
                msg = message(subject=eval_data + " Test Notice",
                          text = self.dataset + "/" + self.model_name + " Test results:\n" + "  P: {0:.3f}\n".format(P) + "  R: {0:.3f}\n".format(R) + "  F1: {0:.3f}".format(F1) + " (Current Path: " + os.getcwd() + ")")
                send(msg)  # and send it
            if not test:
                if F1 > self.MATRES_best_micro_F1 or path.exists(self.MATRES_best_PATH) == False:
                    self.MATRES_best_micro_F1 = F1
                    self.MATRES_best_cm = CM
                    ### save model parameters to .pt file ###
                    torch.save(self.model, self.MATRES_best_PATH)
                    return 1
        
        if eval_data == "HiEve":
            # Report the final accuracy for this validation run.
            cr = classification_report(y_gold, y_pred, output_dict = True)
            rst = classification_report(y_gold, y_pred)
            F1_PC = cr['0']['f1-score']
            F1_CP = cr['1']['f1-score']
            F1_coref = cr['2']['f1-score']
            F1_NoRel = cr['3']['f1-score']
            F1_PC_CP_avg = (F1_PC + F1_CP) / 2.0
            print(rst)
            print("  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg))
            if test:
                #print("  rst:", file = self.file)
                #print(rst, file = self.file)
                #print("  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg), file = self.file)
                msg = message(subject=eval_data + " Test Notice", text = self.dataset + "/" + self.model_name + " Test results:\n" + "  F1_PC_CP_avg: {0:.3f}".format(F1_PC_CP_avg) + " (Current Path: " + os.getcwd() + ")")
                send(msg)  # and send it
            if not test:
                if F1_PC_CP_avg > self.HiEve_best_F1 or path.exists(self.HiEve_best_PATH) == False:
                    self.HiEve_best_F1 = F1_PC_CP_avg
                    self.HiEve_best_prfs = rst
                    torch.save(self.model, self.HiEve_best_PATH)
                    return 1
        return 0