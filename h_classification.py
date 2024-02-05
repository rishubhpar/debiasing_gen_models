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
import time
from torch.autograd import Variable

# Softmax
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(512*8*8, 2) for i in range(49)])
    
    def forward(self,x, t):
        x = x.reshape(x.shape[0],-1)
        x = self.linears[t](x)
        return x

# Sigmoid
class Linear1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(512*8*8, 1)) for i in range(49)])
    
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
    # print(t)
    array = [0, 20, 40, 61, 81, 101, 122, 142, 163, 183, 203, 224, 244, 265, 285, 305, 326, 346, 366, 387, 407, 428, 448, 468, 489, 509, 530, 550, 570, 591, 611, 632, 652, 672, 693, 713, 733, 754, 774, 795, 815, 835, 856, 876, 897, 917, 937, 958, 978, 999]
    return array.index(t)
    # if array.index(t)%5==0:
    #     return array.index(t)//5
    # else:
    #     return -1


#   #
#   # 1 logit
#   #

# This is for normal classification guidance. Class index is the index of the class to be guided towards
@torch.enable_grad()
def gradients(x, t, male = 1, eyeglasses = 1, attribute_list = [0,1,0,0], scale = [1500]):
    
    timestep = get_timetstep(t) 
    if timestep ==-1:return  None

    temperature = 8
    male = int(male)
    eyeglasses = int(eyeglasses)

    softmax = torch.nn.Softmax(dim = -1)
    x = x.detach().requires_grad_()
    
    model_paths = ['classification_checkpoints/new_model/eyeglasses_reall.pt', 'classification_checkpoints/new_model/gender.pt', 'classification_checkpoints/new_model/race.pt', '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Classifier_guidance/checkpoint_pose/pose.pt']
   
    indices = [i for i in range(len(attribute_list)) if attribute_list[i] == 1]

    # Single attribute
    if len(indices)==1:
        model_gender = make_model(model_paths[indices[0]])
        
        start_time = time.time()
        logits = model_gender(x,timestep)/temperature
        logits = F.softmax(logits, dim=1)
        loss = -(logits[:,male].sum())

        gradients = torch.autograd.grad(loss*300, x)


        end_time = time.time()
        with open("timing.txt", 'a') as file1:
            file1.write(str(end_time-start_time))
            file1.write("\n")
        
        return gradients[0]*scale[0]
    

    elif len(indices)==2:
        # print("two", indices)
        model_1 = make_model(model_paths[indices[0]])
        model_2 = make_model(model_paths[indices[1]])

        logits_1 = model_1(x,timestep)/temperature
        logits_1 = F.softmax(logits_1, dim=1)

        logits_2 = model_2(x,timestep)/temperature
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

        gradients1 = torch.autograd.grad(loss1*300, x)[0]
        gradients2 = torch.autograd.grad(loss2*300, x)[0]
        
        return gradients1*scale[0] + gradients2*scale[1]








        
    
# def pose(x,t, scale = [1500]):
#     path = '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Classifier_guidance/checkpoint_pose/imagenet_dog.pt'
#     direction = torch.load(path)

#     timestep = get_timetstep(t) 
#     if timestep ==-1:return  None
#     return (direction[timestep:timestep+1]*scale[0]).to(x.device)


# def flipping_imagenet(x,t, scale = [1500]):
#     path = '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/dataset/Imagenet/h_vectors/10_right.pt'
#     direction = torch.load(path)

#     timestep = get_timetstep(t) 
#     if timestep ==-1:return  None

#     grads = []
#     # scales = np.arange(0,1.1,0.1)
#     # for i in scales:
#     #     grads.append(direction[timestep:timestep+1]*i)
#     # grads =torch.cat(grads).to(x.device)

#     # scales = np.arange(0,1.1,0.1)
#     # for i in scales:
#     #     grads.append(direction[timestep:timestep+1]*1)
#     # grads =torch.cat(grads).to(x.device)

#     # return grads
#     return direction[timestep].to(x.device)



# def pose_gif(x,t, scale = [1500]):
#     path = '/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/rishubh/abhijnya/Classifier_guidance/checkpoint_pose/imagenet_dog.pt'
#     direction = torch.load(path)

#     timestep = get_timetstep(t) 
#     if timestep ==-1:return  None

#     grads = []
#     scales = np.arange(-5,5.5,0.5)
#     for i in scales:
#         grads.append(direction[timestep:timestep+1]*i)
#     grads =torch.cat(grads).to(x.device)

#     return grads




# # This is to make the majority fraction as 0.5
# @torch.enable_grad()
# def gradients_fraction(x, t, path):
#     scaling = 1e6
#     x = x.detach().requires_grad_()
#     model = make_model(path)
#     # logits = F.softmax(model(x,get_timetstep(t)), dim=1)
#     logits = model(x,get_timetstep(t))

#     female = (1-logits[logits<0.5]).sum()
#     male = (logits[logits>0.5].sum())
#     total = female+male

#     female = female/total
#     male = male/total
#     print(female, male)
#     print("numb of fem", len(logits[logits<0.5]), len(logits[logits>0.5]))
    

#     if female>male:
#         loss = female/total
#         loss = abs(loss-0.5)
#         gradients = torch.autograd.grad(loss*scaling, x)[0]
#     elif male>female:
#         loss = male/total
#         loss = abs(loss-0.5)
#         gradients = torch.autograd.grad(loss*scaling, x)[0]
#     else: return None, None

#     print("gradss",gradients.mean())
#     males = len(logits>0.5)
#     return gradients, males 



#   #
#   # 2 logits
#   #

# @torch.enable_grad()
# def gradients_point_five(x, t, path):
#     temperature = 8
#     x = x.detach().requires_grad_()
    
#     timestep = get_timetstep(t) 
#     if timestep ==-1:return  None, None

#     model_gender = make_model(path)
#     logits = model_gender(x,timestep)/temperature
#     logits = F.softmax(logits, dim=1)

#     # Batch wise entropy maximization
#     loss_gender = torch.mean(logits, dim = 0)
#     loss_gender = (loss_gender[0]*torch.log(loss_gender[0]/0.2)) + (loss_gender[1]*torch.log(loss_gender[1]/0.8))

#     # Eyeglasses
#     model_glasses = make_model('./classification_checkpoints/eyeglasses_linear.pt')


#     logits = model_glasses(x,timestep)/temperature
#     logits = F.softmax(logits, dim=1)

    # # Batch wise entropy maximization
    # loss = torch.mean(logits, dim = 0)
    # loss = (loss[0]*torch.log(loss[0]/0.8)) + (loss[1]*torch.log(loss[1]/0.2))


    # loss = loss + loss_gender

    # gradients = torch.autograd.grad(loss*150, x)[0]
    
    # # males = F.softmax(logits, dim = 1)[:,1]
    # females = logits[:,0]>logits[:,1]
    # print(females.sum())
    # return gradients, females



#   # This minimizes the entropy 
# @torch.enable_grad()
# def gradients_point_five(x, t, path):
#     temperature = 8
#     x = x.detach().requires_grad_()
#     model = make_model(path)

#     timestep = get_timetstep(t) 
#     if timestep ==-1:return  None, None

#     logits = model(x,timestep)/temperature
#     logits = F.softmax(logits, dim=1)

#     # # Element wise entropy minimization
#     # loss1 = torch.sum(logits*torch.log(logits), dim=1)
#     # # print(loss1.shape)
#     # loss1 = -(loss1.mean())

#     # Batch wise entropy maximization
#     loss = torch.mean(logits, dim = 0)
#     # loss = (loss*torch.log(loss)).sum()
#     loss = (loss[0]*torch.log(loss[0]/0.2)) + (loss[1]*torch.log(loss[1]/0.8))


#     # loss = loss + loss1

#     gradients = torch.autograd.grad(loss*150, x)[0]
    
#     # males = F.softmax(logits, dim = 1)[:,1]
#     females = logits[:,0]>logits[:,1]
#     print(females.sum())
#     return gradients, females


#   # This is for normal classification guidance. Class index is the index of the class to be guided towards
# @torch.enable_grad()
# def gradients_og(x, t):
#     print("taking new classifier")
#     x = x.detach().requires_grad_()
#     model = make_model('/mnt/data/rishubh/abhijnya/Classifier_guidance/classification_checkpoints/new_model/race.pt')
#     logits = model(x,get_timetstep(t))
#     loss = -logits[:,1].sum()
#     gradients = torch.autograd.grad(loss, x)
#     return gradients[0]


# # This is to make the majority fraction as 0.5
# @torch.enable_grad()
# def gradients_fraction(x, t, path):
#     x = x.detach().requires_grad_()
#     model = make_model(path)
#     logits = F.softmax(model(x,get_timetstep(t)), dim=1)

#     female = logits[:,0].sum()
#     male = logits[:,1].sum()
#     total = female+male

#     if female>male:
#         loss = female/total
#         loss = abs(loss-0.5)
#         print(loss)
#         gradients = torch.autograd.grad(loss, x)[0]
#     elif male>female:
#         loss = male/total
#         loss = abs(loss-0.5)
#         gradients = torch.autograd.grad(loss, x)[0]
#     else: return None, None

#     males = F.softmax(logits, dim = 1)[:,1]
#     return gradients, males 


#   # This is using mse between average of male/female logits and 0.5
# @torch.enable_grad()
# def gradients_point_five(x, t, path):
#     x = x.detach().requires_grad_()
#     model = make_model(path)
#     mse = nn.MSELoss()
#     logits = F.softmax(model(x,get_timetstep(t)), dim=1)

#     truth = torch.full((logits.shape[0], 1), 0.5).cuda()
#     loss = torch.log10(mse(logits[:,0].mean(), truth) + mse(logits[:,1].mean(), truth))

#     gradients = torch.autograd.grad(loss*150, x)[0]

#     males = F.softmax(logits, dim = 1)[:,1]
#     return gradients, males 


#   # This is code for selecting extra majority elements and changing them to minority
# @torch.enable_grad()
# def gradients_equal(h,t,path):

#     timestep = get_timetstep(t)
#     model = make_model(path)
#     batch_size = h.shape[0]

#     logits = model(h,timestep)
#     val, index = torch.max(logits,axis=1)
#     # Finding the number of males and females
#     zero = (index==0).sum()
#     one = batch_size - zero
#     # If females are greater
#     if zero>one:
#         # Getting the number of females to convert to males (50% - males)
#         difference = int((batch_size/2) - one)
#         print("females are greater by ", difference)
#         diff = logits[:,0]-logits[:,1]
#         # Index of each h vector's logit
#         x = torch.tensor([i for i in range(batch_size)]).cuda()
#         input = torch.stack((diff,x),dim=1)
#         # Removing all males
#         input = input[(input >= 0).all(axis=1)]
#         # Getting the indeces (by sorting in ascending order of difference of logit values)
#         indices = torch.sort(input[input[:, 0].sort()[1]][:difference][:,1].long())[0]
#         # Calculating gradient only for those indices
#         temp = h[[indices]]
#         temp = temp.requires_grad_()
#         loss = model(temp,timestep)
#         # Loss in favour of males
#         loss = -loss[:,1].sum()
#         gradients = torch.autograd.grad(loss, temp)[0]
#         return gradients, indices

#     elif one>zero:
#         # Getting the number of males to convert to females (50% - males)
#         difference = int((batch_size/2) - zero)
#         print("males are greater by ", difference)
#         diff = logits[:,1]-logits[:,0]
#         # Index of each h vector's logit
#         x = torch.tensor([i for i in range(batch_size)]).cuda()
#         input = torch.stack((diff,x),dim=1)
#         # Removing all females
#         input = input[(input >= 0).all(axis=1)]
#         # Getting the indeces (by sorting in ascending order of difference of logit values)
#         indices = torch.sort(input[input[:, 0].sort()[1]][:difference][:,1].long())[0]
#         # Calculating gradient only for those indices
#         temp = h[[indices]]
#         temp = temp.requires_grad_()
#         loss = model(temp,timestep)
#         # Loss in favour of females
#         loss = -loss[:,0].sum()
#         gradients = torch.autograd.grad(loss, temp)[0]
#         return gradients, indices
    
#     else:
#         print("50 50 already")
#         return None, None


# # Changing not only the extra h vectors but also the minority class
# @torch.enable_grad()
# def gradients_equal_extra(h,t,path):

#     timestep = get_timetstep(t)
#     model = make_model(path)
#     batch_size = h.shape[0]

#     logits = model(h,timestep)
#     val, index = torch.max(logits,axis=1)
#     # Finding the number of males and females
#     zero = (index==0).sum()
#     one = batch_size - zero
#     # If females are greater
#     if zero>one:
#         # Getting the number of females to convert to males (50% - males)
#         difference = int((batch_size/2) - one)
#         print("females are greater by ", difference)

#         diff = logits[:,0]-logits[:,1]
#         # Index of each h vector's logit
#         x = torch.tensor([i for i in range(batch_size)]).cuda()
#         input = torch.stack((diff,x),dim=1)
#         # Removing all males
#         input = input[(input >= 0).all(axis=1)]
        
#         # All females
#         fems = input[input[:, 0].sort()[1]]
        
#         # All males (x-fems)
#         compareview = fems[:,1].long().expand(x.shape[0], fems[:,1].long().shape[0]).T
#         male_index = x[(compareview != x).T.prod(1)==1]

#         # Getting the indeces (by sorting in ascending order of difference of logit values)
#         changing_indices = fems[:difference][:,1].long()
#         # Total males (all males + extra)
#         male_index = torch.sort(torch.cat([male_index,changing_indices]))[0]

#         # female_index = fems[difference:][:,1].long()

#         # Loss for males only
#         temp = h[[male_index]]
#         temp = temp.requires_grad_()
#         loss = model(temp,timestep)
#         # Loss in favour of females
#         loss = -loss[:,1].sum()
#         gradients = torch.autograd.grad(loss, temp)[0]
#         return gradients, male_index

#     # If males are greater
#     elif one>zero:
#         # Getting the number of males to convert to females (50% - males)
#         difference = int((batch_size/2) - zero)
#         print("males are greater by ", difference)

#         diff = logits[:,1]-logits[:,0]
#         # Index of each h vector's logit
#         x = torch.tensor([i for i in range(batch_size)]).cuda()
#         input = torch.stack((diff,x),dim=1)
#         # Removing all females
#         input = input[(input >= 0).all(axis=1)]

#         # All Males
#         fems = input[input[:, 0].sort()[1]]

#         # All Females
#         compareview = fems[:,1].long().expand(x.shape[0], fems[:,1].long().shape[0]).T
#         male_index = x[(compareview != x).T.prod(1)==1]

#         # Getting the indeces (by sorting in ascending order of difference of logit values)
#         changing_indices = fems[:difference][:,1].long()
#         # Total females (all females + extra)
#         male_index = torch.sort(torch.cat([male_index,changing_indices]))[0]

#         # female_index = fems[difference:][:,1].long()

#         # Calculating gradient only for those indices
#         temp = h[[male_index]]
#         temp = temp.requires_grad_()
#         loss = model(temp,timestep)
#         # Loss in favour of females
#         loss = -loss[:,0].sum()
#         gradients = torch.autograd.grad(loss, temp)[0]
#         return gradients, male_index

#     else:
#             print("50 50 already")
#             return None, None