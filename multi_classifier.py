from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch import optim 
from torchvision import transforms, utils, models
import copy
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
# device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
import torchvision.transforms.functional as Funct
from torch.autograd import Variable
import matplotlib.pyplot as plt



# Softmax
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(512*8*8, 2) for i in range(49)])
    
    def forward(self,x, t):
        x = x.reshape(x.shape[0],-1)
        x = self.linears[t](x)
        return x

    
def make_model(path):
    model = Linear().cuda()
    model.load_state_dict(torch.load(path, map_location='cuda:0'))
    model.eval()

    return model



def get_timetstep(t):
    array = [0, 20, 40, 61, 81, 101, 122, 142, 163, 183, 203, 224, 244, 265, 285, 305, 326, 346, 366, 387, 407, 428, 448, 468, 489, 509, 530, 550, 570, 591, 611, 632, 652, 672, 693, 713, 733, 754, 774, 795, 815, 835, 856, 876, 897, 917, 937, 958, 978, 999]
    return array.index(t)
    # if array.index(t)%5==0:
    #     return array.index(t)//5
    # else:
    #     return -1

@torch.enable_grad()
def gradients_point_five(x, t, male = 0.5, eyeglasses = 0.5, smile = 0.5, guidance_loss='chi_without_5', attribute_list = [0,1,0,0], scale=[1500,700]):
    temperature = 8
    x = x.detach().requires_grad_()
    
    timestep = get_timetstep(t) 
    if timestep ==-1:return  None

    model_paths = ['classification_checkpoints/new_model/eyeglasses_reall.pt', 'classification_checkpoints/new_model/gender.pt', 'classification_checkpoints/new_model/race.pt', 'classification_checkpoints/new_model/smile.pt']
   
    indices = [i for i in range(len(attribute_list)) if attribute_list[i] == 1]


    # Single attribute
    if len(indices)==1:
        # print("single")
        model_gender = make_model(model_paths[indices[0]])

        logits = model_gender(x,timestep)/temperature
        logits = F.softmax(logits, dim=1)

        if guidance_loss == 'kl_without_5':
            # print(guidance_loss)
            loss_gender = torch.mean(logits, dim = 0)
            loss = (loss_gender[0]*torch.log(loss_gender[0]/(1-male))) + (loss_gender[1]*torch.log(loss_gender[1]/male))

        elif guidance_loss == 'kl_with_5':
            # print(guidance_loss)
            # KL itself, but only using the values>0.5
            fem = torch.sum(logits[:,0][logits[:,0]>0.5])/len(logits)
            mal = torch.sum(logits[:,1][logits[:,1]>0.5])/len(logits)
            # Considering individual attributes
            loss = (fem*torch.log(fem/(1-male))) + (mal*torch.log(mal/male))

        elif guidance_loss == 'chi_without_5':
            # print(guidance_loss)
            #  # Chi square
            loss_gender = torch.mean(logits, dim = 0)
            loss = ((loss_gender[0]-(1-male))**2)/(1-male) + ((loss_gender[1]-male)**2)/male


        elif guidance_loss == 'chi_with_5':
            # print(guidance_loss)
            # Chi square w >0.5
            fem = torch.sum(logits[:,0][logits[:,0]>0.5])/len(logits)
            mal = torch.sum(logits[:,1][logits[:,1]>0.5])/len(logits)
            loss = ((fem-(1-male))**2)/(1-male) + ((mal-male)**2)/male

        else:
            print("wrong loss, exiting")
            exit()

        gradients = torch.autograd.grad(loss*300, x)[0]



        return gradients*scale[0]



    # Two attributes
    elif len(indices)==2:
        # print("two", indices)
        model_1 = make_model(model_paths[indices[0]])
        model_2 = make_model(model_paths[indices[1]])

        logits_1 = model_1(x,timestep)/temperature
        logits_1 = F.softmax(logits_1, dim=1)

        logits_2 = model_2(x,timestep)/temperature
        logits_2 = F.softmax(logits_2, dim=1)

        if guidance_loss == 'kl_without_5':
            # print(guidance_loss)
            logits_1 = torch.mean(logits_1, dim = 0)
            loss1 = (logits_1[0]*torch.log(logits_1[0]/(1-male))) + (logits_1[1]*torch.log(logits_1[1]/male))

            logits_2 = torch.mean(logits_2, dim = 0)
            loss2 = (logits_2[0]*torch.log(logits_2[0]/(1-male))) + (logits_2[1]*torch.log(logits_2[1]/male))

        elif guidance_loss == 'kl_with_5':
            # print(guidance_loss)
            # KL itself, but only using the values>0.5
            fem = torch.sum(logits_1[:,0][logits_1[:,0]>0.5])/len(logits_1)
            mal = torch.sum(logits_1[:,1][logits_1[:,1]>0.5])/len(logits_1)
            # Considering individual attributes
            loss1 = (fem*torch.log(fem/(1-male))) + (mal*torch.log(mal/male))

            fem = torch.sum(logits_2[:,0][logits_2[:,0]>0.5])/len(logits_2)
            mal = torch.sum(logits_2[:,1][logits_2[:,1]>0.5])/len(logits_2)
            # Considering individual attributes
            loss2 = (fem*torch.log(fem/(1-male))) + (mal*torch.log(mal/male))


        elif guidance_loss == 'chi_without_5':
            # print(guidance_loss)
            #  # Chi square
            logits_1 = torch.mean(logits_1, dim = 0)
            loss1 = ((logits_1[0]-(1-male))**2)/(1-male) + ((logits_1[1]-male)**2)/male

            logits_2 = torch.mean(logits_2, dim = 0)
            loss2 = ((logits_2[0]-(1-male))**2)/(1-male) + ((logits_2[1]-male)**2)/male


        elif guidance_loss == 'chi_with_5':
            # print(guidance_loss)
            # Chi square w >0.5
            fem = torch.sum(logits_1[:,0][logits_1[:,0]>0.5])/len(logits_1)
            mal = torch.sum(logits_1[:,1][logits_1[:,1]>0.5])/len(logits_1)
            loss1 = ((fem-(1-male))**2)/(1-male) + ((mal-male)**2)/male

            fem = torch.sum(logits_2[:,0][logits_2[:,0]>0.5])/len(logits_2)
            mal = torch.sum(logits_2[:,1][logits_2[:,1]>0.5])/len(logits_2)
            loss2 = ((fem-(1-male))**2)/(1-male) + ((mal-male)**2)/male

        else:
            print("wrong loss, exiting")
            exit()

        gradients1 = torch.autograd.grad(loss1*300, x)[0]
        gradients2 = torch.autograd.grad(loss2*300, x)[0]
        
        return gradients1*scale[0] + gradients2*scale[1]



    else:
        print("wrong attribute list")
        exit()
        
