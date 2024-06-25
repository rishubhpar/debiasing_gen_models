from audioop import reverse
from genericpath import isfile
from glob import glob
from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.nn import normalization
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from losses.clip_loss import CLIPLoss
import random
import copy
import matplotlib.pyplot as plt


from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from utils.text_dic import SRC_TRG_TXT_DIC
from losses import id_loss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from datasets.imagenet_dic import IMAGENET_DIC



class Asyrp(object):
    def __init__(self, args, config, device=None):
        # ----------- predefined parameters -----------#
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = False # it will be changed in load_pretrained_model()

        # # ----------- Editing txt -----------#
        # if self.args.edit_attr is None:
        #     # self.src_txts = self.args.src_txts
        #     # self.trg_txts = self.args.trg_txts
        # elif self.args.edit_attr == "attribute":
        #     pass
        # else:
        #     # print(SRC_TRG_TXT_DIC)
        #     # self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
        #     # self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


    def load_pretrained_model(self):

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset in ["CelebA_HQ", "CUSTOM", "CelebA_HQ_Dialog"]:
            # Idk? maybe SDE2
            url = "/mnt/data/rishubh/abhijnya/Classifier_guidance/pretrained/celeba_hq.ckpt"
            # url = "https://drive.google.com/file/d/1cSEIFLjOAyiabGnbIa2ZhelOizRQLerM/view"
            # url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MetFACE","CelebA_HQ_P2"]:
            # get the model ["FFHQ", "AFHQ", "MetFACE"] from 
            # https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            # reference : ILVR (https://arxiv.org/abs/2108.02938), P2 weighting (https://arxiv.org/abs/2204.00227)
            # reference github : https://github.com/jychoi118/ilvr_adm , https://github.com/jychoi118/P2-weighting 

            # get the model "IMAGENET" from
            # https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
            # reference : ADM (https://arxiv.org/abs/2105.05233)
            pass
        else:
            # if you want to use LSUN-horse, LSUN-cat -> https://github.com/openai/guided-diffusion
            # if you want to use CUB, Flowers -> https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            raise ValueError
        if self.config.data.dataset in ["CelebA_HQ", "LSUN", "CelebA_HQ_Dialog"]:
            model = DDPM(self.config) 
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(url, map_location=self.device)
                # init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            self.learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset) #Get_h(self.config, model="i_DDPM", layer_num=self.args.get_h_num) #
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["MetFACE", "CelebA_HQ_P2"]:
            model = guided_Diffusion(self.config.data.dataset)
            init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            print("Model loaded", self.config.data.dataset)
            self.learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt, strict=False)

        return model

    
    @torch.no_grad()
    def save_image(self, model, x_lat_tensor, seq_inv, seq_inv_next, save_process_origin = False, get_delta_hs=False,
                    folder_dir="", file_name="", hs_coeff=(1.0,1.0),):
    

        # if save_process_origin or save_process_delta_h:
        #     os.makedirs(os.path.join(folder_dir,file_name), exist_ok=True)

        # process_num = int(save_x_origin) + (len(hs_coeff) if isinstance(hs_coeff, list) else 1)
        # print("processsssssss number", process_num)
        # print(len(seq_inv)*(process_num))
        # print()
        
        with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
            x_list = []

            # if save_x0:
            #     if x0_tensor is not None:
            #         x_list.append(x0_tensor.to(self.device))
            
            # if save_x_origin:
            labels = None
    
            # No delta h
            x = x_lat_tensor.clone().to(self.device)

            for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                t = (torch.ones(1) * i).to(self.device)
                t_next = (torch.ones(1) * j).to(self.device)
                
                x, x0_t, grads, _  = denoising_step(x, t=t, t_next=t_next, models=model,
                                logvars=self.logvar,
                                sample = self.args.sample,
                                male = self.args.male,
                                eyeglasses = self.args.eyeglasses,
                                scale = self.args.scale,
                                timestep_list = self.args.timestep_list,
                                # usefancy = self.args.usefancy,
                                # gamma_factor = self.args.gamma_factor,
                                # guidance_loss = self.args.guidance_loss,
                                attribute_list = self.args.attribute_list,
                                vanilla_generation = self.args.vanilla_generation,
                                universal_guidance=self.args.universal_guidance,
                                sampling_type= self.args.sample_type,
                                b=self.betas,
                                learn_sigma=self.learn_sigma,
                                eta=1.0
                                )
                progress_bar.update(1)


                # if save_process_origin:
                #     output = torch.cat([x, x0_t], dim=0)
                #     output = (output + 1) * 0.5
                #     grid = tvu.make_grid(output, nrow=self.args.bs_train, padding=1)
                #     tvu.save_image(grid, os.path.join(folder_dir, file_name, f'origin_{int(t[0].item())}.png'), normalization=True)

            x_list.append(x)
                


            # if self.args.pass_editing:
            #     pass


        x = torch.cat(x_list, dim=0)


        x = (x + 1) * 0.5

        # print(x.shape)
        for image_index, image_tosave in enumerate(x):
            tvu.save_image(image_tosave, os.path.join(folder_dir, f'{file_name}_{image_index}_ngen{self.args.n_train_step}.png'), normalization=True)
          
          
    # test
    @torch.no_grad()
    def run_test(self):
        print("Running Test")

        self.set_t_edit_t_addnoise(LPIPS_th=self.args.lpips_edit_th, 
                                                            LPIPS_addnoise_th=self.args.lpips_addnoise_th,
                                                            return_clip_loss=False)

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s+1e-6) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        # seq_test: list of steps from 0 till 1000
        print("loading model pretrained")
        # ----------- Model -----------#
        model = self.load_pretrained_model()

        # if self.args.train_delta_block:
        #     model.setattr_layers(self.args.get_h_num)
        #     print("Setattr layers")
            

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        print("model moved to device")

        # ----------- Pre-compute -----------#
        print("Prepare identity latent...")


        if self.args.just_precompute:
            # if you just want to precompute, you can stop here.  i.e. just finding the inversion latents and stop
            img_lat_pairs_dic = self.precompute_pairs(model, self.args.save_precomputed_images)
            print("Pre-computed done.")
            return
        else:
            print("using random noise")
            img_lat_pairs_dic = self.random_noise_pairs(model)


        print("number of noise vectors", len(img_lat_pairs_dic))
        if self.args.target_image_id:
            self.args.target_image_id = self.args.target_image_id.split(" ")
            self.args.target_image_id = [int(i) for i in self.args.target_image_id]

        # Unfortunately, ima_lat_pairs_dic does not match with batch_size
        x_lat_tensor = None
        # x0_tensor = None
        model.eval()
        
        # Test set

        # if self.args.do_test:
        print("inside testing set")

        x_lat_tensor = None

        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['test']):
            # print(x0.shape)
            if self.args.target_image_id:
                # assert self.args.bs_train == 1, "target_image_id is only supported for batch_size == 1"
                if not step in self.args.target_image_id:
                    continue

            if self.args.start_image_id > step:
                continue

            if x_lat_tensor is None:
                x_lat_tensor = x_lat
                # x0_tensor = x0
            else:
                x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                # x0_tensor = torch.cat((x0_tensor, x0), dim=0)
            # if (step+1) % self.args.bs_train != 0:
            #     continue
            
            self.save_image(model, x_lat_tensor, seq_test, seq_test_next,
                                        folder_dir=self.args.test_image_folder,
                                        save_process_origin=self.args.save_process_origin,
                                        file_name=f'test_{step}_{self.args.n_iter - 1}'
                                        )
                                    
            if (step+1)*self.args.bs_test >= self.args.n_test_img:
                break
            x_lat_tensor = None


    # ----------- Pre-compute -----------#
    # @torch.no_grad()
    def precompute_pairs(self, model):
    
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        for mode in ['test']:
            img_lat_pairs = []

        
            print("no path or recompute")

            # if self.config.data.category == 'CUSTOM':
                
                # print("custom:", self.args.custom_train_dataset_dir)
                # DATASET_PATHS["custom_train"] = self.args.custom_train_dataset_dir


            DATASET_PATHS["custom_test"] = os.path.join(self.args.test_path_one)
            print("path to generate:", DATASET_PATHS["custom_test"])

            test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
            loader_dic = get_dataloader(test_dataset, num_workers=self.config.data.num_workers, shuffle=False)
            loader = loader_dic[mode]
            
            # if self.args.save_process_origin:
            #     save_process_folder = os.path.join(self.args.image_folder, f'inversion_process')
            #     if not os.path.exists(save_process_folder):
            #         os.makedirs(save_process_folder)


            for step, (img, label) in enumerate(loader):
                if os.path.exists(os.path.join(self.args.savepath, f'{label[0].split(".")[0]}.pt')):
                    continue
                
                if (mode == "test" and step == self.args.n_test_img):
                    break
                x0 = img.to(self.config.device)
                # if save_imgs:
                #     tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                
                with torch.no_grad():
                    h_vector = []

                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, _, _, h = denoising_step( x, t=t, t_next=t_prev, models=model,
                                            logvars=self.logvar,
                                            sampling_type='ddim',
                                            b=self.betas,
                                            eta=0,
                                            learn_sigma=self.learn_sigma,
                                            )
                            
                            for i in range(len(h)):
                                h_vector.append(h.detach().cpu())
                                
                            progress_bar.update(1)
                    

                    h_vector = torch.cat(h_vector, dim=0)
                    torch.save(h_vector, os.path.join(self.args.savepath, f'{label[0].split(".")[0]}.pt'))

    # ----------- Get random latent -----------#
    @torch.no_grad()
    def random_noise_pairs(self, model):

        print("Prepare random latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        test_lat = []

        for i in range(self.args.n_test_img//self.args.bs_test):
            lat = torch.randn((self.args.bs_test, self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        if  self.args.n_test_img%self.args.bs_test!=0:
            lat = torch.randn((self.args.n_test_img%self.args.bs_test, self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        img_lat_pairs_dic['test'] = test_lat

            

        return img_lat_pairs_dic

    @torch.no_grad()
    def set_t_edit_t_addnoise(self, LPIPS_th=0.33, LPIPS_addnoise_th=0.1, return_clip_loss=False):

        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)
