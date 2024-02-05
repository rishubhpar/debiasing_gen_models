from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from  torch import optim 
from torchvision import transforms, utils, models, datasets
import copy
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as Funct
from torch.autograd import Variable
import torchvision.utils as tvu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import subprocess
import matplotlib.pyplot as plt
import argparse 
from torchvision.models import resnet50, ResNet50_Weights

 

# -------------------------------------------------------------------------------------
size_of_image = (256, 256)
test_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()

class Dc_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(2048,512)
        self.linear2=nn.Linear(512,2)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

def make_model(path):
    model =  resnet50(weights=None)
    model_ = Dc_model()
    model.fc = model_
    model = model.cuda()
    model.load_state_dict(torch.load(path, map_location = "cuda:0"))
    model.eval()
    return model  


# -------------------------------------------------------------------------------------

@torch.enable_grad()
def get_image(et, xt, at, male, scale, attribute_list):
    temperature = 8
    male = int(male)
    # Getting predicted x0
    xt = xt.detach().requires_grad_()
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    x0_t = (x0_t + 1) * 0.5
    x0_t = test_transforms(x0_t)

    model_paths = ['/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Resnet18/checkpoints_image_classifier/Eyeglasses_30k.pt', '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Resnet18/checkpoints_image_classifier/Male_30k.pt', '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Resnet18/checkpoints_image_classifier/Race_30k.pt', '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Resnet18/checkpoint_different_domain/sketch.pt']
    # model_paths = ['/data/abhijnya/Resnet18/image_space_guidance_ckpt/eyeglasses_gt.pt', '/data/abhijnya/Resnet18/image_space_guidance_ckpt/gender.pt', '/data/abhijnya/Resnet18/image_space_guidance_ckpt/race.pt', '/data/abhijnya/Resnet18/image_space_guidance_ckpt/smile.pt']
    

    indices = [i for i in range(len(attribute_list)) if attribute_list[i] == 1]

    if len(indices)==1:
        # print("Inside, ", model_paths[indices[0]])
        model = make_model(model_paths[indices[0]])

        y_pred = model(x0_t.cuda())/temperature
        y_pred = F.softmax(y_pred, dim=1)
        loss = -(y_pred[:,male].sum())
        gradients = torch.autograd.grad(loss, xt)[0]

        return gradients*scale[0]

    elif len(indices)==2:
        # print("two", indices)
        model_1 = make_model(model_paths[indices[0]])
        model_2 = make_model(model_paths[indices[1]])

        logits_1 = model_1(x0_t.cuda())/temperature
        logits_1 = F.softmax(logits_1, dim=1)

        logits_2 = model_2(x0_t.cuda())/temperature
        logits_2 = F.softmax(logits_2, dim=1)

        if male==0:
            # print("00")
            loss1 = -(logits_1[:,0].sum())
            loss2 = -(logits_2[:,0].sum())

        elif male==1:
            # print("01")
            loss1 = -(logits_1[:,0].sum())
            loss2 = -(logits_2[:,1].sum())

        elif male==2:
            # print("10")
            loss1 = -(logits_1[:,1].sum())
            loss2 = -(logits_2[:,0].sum())

        elif male==3:
            # print("11")
            loss1 = -(logits_1[:,1].sum())
            loss2 = -(logits_2[:,1].sum())

        else:
            print("wrong configuration")
            exit()

        gradients1 = torch.autograd.grad(loss1*300, xt, retain_graph=True)[0]
        gradients2 = torch.autograd.grad(loss2*300, xt)[0]
        
        return gradients1*scale[0] + gradients2*scale[1]
    


