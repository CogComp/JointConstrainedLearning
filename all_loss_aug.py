import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import pickle
import random
import sys

class transition_loss_(torch.nn.Module):
    def __init__(self):
        super(transition_loss_, self).__init__()
        self.zero = Variable(torch.zeros(1), requires_grad=False)
        self.zero = self.zero.cuda()

    def forward(self, log_y_alpha, log_y_beta, log_y_gamma, alpha_index, beta_index, gamma_index, label_weight = None):
        if label_weight is None:
            label_w = 1
        else:
            label_w = label_weight.sum() / label_weight[alpha_index * 16 + beta_index * 4 + gamma_index] / 64.0
        return torch.max(self.zero, log_y_alpha[:, alpha_index] + log_y_beta[:, beta_index] - log_y_gamma[:, gamma_index]) * label_w
    
class transition_loss_not_(torch.nn.Module):
    def __init__(self):
        super(transition_loss_not_, self).__init__()
        self.zero = Variable(torch.zeros(1), requires_grad=False)
        self.one = Variable(torch.ones(1), requires_grad=False)
        self.zero = self.zero.cuda()
        self.one = self.one.cuda()

    def forward(self, log_y_alpha, log_y_beta, log_y_gamma, alpha_index, beta_index, gamma_index):
        very_small = 1e-8
        log_not_y_gamma = (self.one - log_y_gamma.exp()).clamp(very_small).log()
        return torch.max(self.zero, log_y_alpha[:, alpha_index] + log_y_beta[:, beta_index] - log_not_y_gamma[:, gamma_index])

class transitivity_loss_H_(torch.nn.Module):
    def __init__(self):
        super(transitivity_loss_H_, self).__init__()
        
    def forward(self, alpha_logits, beta_logits, gamma_logits, label_weight_H = None):
        log_y_alpha = nn.LogSoftmax(1)(alpha_logits[:, 0:4])
        log_y_beta = nn.LogSoftmax(1)(beta_logits[:, 0:4])
        log_y_gamma = nn.LogSoftmax(1)(gamma_logits[:, 0:4])

        transition_loss = transition_loss_()
        transition_loss_not = transition_loss_not_()
        loss = transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 1, 1, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 2, 2, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 0, 0, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 0, 0, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 2, 0, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 1, 1, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 2, 1, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 3, 3, label_weight_H)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 2, 3, label_weight_H)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 0)
        return loss
    
class transitivity_loss_T_(torch.nn.Module):
    def __init__(self):
        super(transitivity_loss_T_, self).__init__()
        
    def forward(self, alpha_logits, beta_logits, gamma_logits, label_weight_T = None):
        log_y_alpha = nn.LogSoftmax(1)(alpha_logits[:, 0:4])
        log_y_beta = nn.LogSoftmax(1)(beta_logits[:, 0:4])
        log_y_gamma = nn.LogSoftmax(1)(gamma_logits[:, 0:4])

        transition_loss = transition_loss_()
        transition_loss_not = transition_loss_not_()
        loss = transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 0, 0, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 2, 0, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 1, 1, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 2, 1, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 0, 0, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 1, 1, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 2, 2, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 3, 3, label_weight_T)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 2, 3, label_weight_T)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 2)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 2)
        
        return loss
    
class cross_category_loss_(torch.nn.Module):
    def __init__(self):
        super(cross_category_loss_, self).__init__()
        
    def forward(self, alpha_logits, beta_logits, gamma_logits):
        log_y_alpha = nn.LogSoftmax(1)(alpha_logits)
        log_y_beta = nn.LogSoftmax(1)(beta_logits)
        log_y_gamma = nn.LogSoftmax(1)(gamma_logits)

        transition_loss = transition_loss_()
        transition_loss_not = transition_loss_not_()
        loss = transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 4)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 4)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 5)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 5)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 4)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 5)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 6, 6)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 7, 7)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 2, 7, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 4)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 4)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 1)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 5)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 5)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 0)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 2)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 6, 2, 6)
        loss += transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 7, 2, 7)
        loss += transition_loss_not(log_y_alpha, log_y_beta, log_y_gamma, 7, 2, 2)
        return loss