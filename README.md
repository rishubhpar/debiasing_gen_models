# 'Balancing Act: Distribution-Guided Debiasing in Diffusion Models', CVPR 2024 

<!--- [![arXiv](https://img.shields.io/badge/arXiv-2110.02711-red)](https://arxiv.org/abs/2210.10960) [![project_page](https://img.shields.io/badge/project_page-orange)](https://kwonminki.github.io/Asyrp/) 


> **Diffusion Models already have a Semantic Latent Space**<br>
> [Mingi Kwon](https://drive.google.com/file/d/1d1TOCA20KmYnY8RvBvhFwku7QaaWIMZL/view?usp=share_link), [Jaeseok Jeong](https://drive.google.com/file/d/14uHCJLoR1AFydqV_neGjl1H2rjN4HBdv/view), [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor) <br>
> Arxiv preprint.
> 
>**Abstract**: <br>
Diffusion models achieve outstanding generative performance in various domains. Despite their great success, they lack semantic latent space which is essential for controlling the generative process. To address the problem, we propose asymmetric reverse process (Asyrp) which discovers the semantic latent space in frozen pretrained diffusion models. Our semantic latent space, named h-space, has nice properties for accommodating semantic image manipulation: homogeneity, linearity, robustness, and consistency across timesteps. In addition, we introduce a principled design of the generative process for versatile editing and quality boosting by quantifiable measures: editing strength of an interval and quality deficiency at a timestep. Our method is applicable to various architectures (DDPM++, iDDPM, and ADM) and datasets (CelebA-HQ, AFHQ-dog, LSUN-church, LSUN-bedroom, and METFACES). 
 

## Description
This repo includes the official Pytorch implementation of **Asyrp**: Diffusion Models already have a Semantic Latent Space.

- **Asyrp** allows using *h-space*, the bottleneck of the U-Net, as a semantic latent space of diffusion models.

![image](https://user-images.githubusercontent.com/33779055/210209549-500e57d1-0a38-4167-a437-f1dcc44b5a5a.png) ![image](https://user-images.githubusercontent.com/33779055/210209586-096ec082-f2d2-4690-84c9-ce0143361069.png) ![image](https://user-images.githubusercontent.com/33779055/210209619-6091bf02-e81b-468f-a2d0-df893040ab66.png)

Edited real images (Top) as `Happy dog` (Bottom). So cute!!





## Getting Started
We recommend running our code using NVIDIA GPU + CUDA, CuDNN.

### Pretrained Models for Asyrp
Asyrp works on the checkpoints of pretrained diffusion models.


| Image Type to Edit |Size| Pretrained Model | Dataset | Reference Repo. 
|---|---|---|---|---
| Human face |256×256| Diffusion (Auto) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [SDEdit](https://github.com/ermongroup/SDEdit)
| Human face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| Human face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [FFHQ](https://arxiv.org/abs/1812.04948) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| Church |256×256| Diffusion (Auto) | [LSUN-Bedroom](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Bedroom |256×256| Diffusion (Auto) | [LSUN-Church](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Dog face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [AFHQ-Dog](https://arxiv.org/abs/1912.01865) | [ILVR](https://github.com/jychoi118/ilvr_adm)
| Painting face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [METFACES](https://arxiv.org/abs/2006.06676) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| ImageNet |256x256| [Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | [ImageNet](https://image-net.org/index.php) | [Guided Diffusion](https://github.com/openai/guided-diffusion)

- The pretrained Diffuson models on 256x256 images in [CelebA-HQ](https://arxiv.org/abs/1710.10196), [LSUN-Church](https://www.yf.io/p/lsun), and [LSUN-Bedroom](https://www.yf.io/p/lsun) are automatically downloaded in the code. (codes from [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP))
- In contrast, you need to download the models pretrained on other datasets in the table and put it in the `./pretrained` directory. 
- You can manually revise the checkpoint paths and names in the `./configs/paths_config.py` file.

- We used CelebA-HQ pretrained model from SDEdit but we found from P2 weighting is better. **We highly recommend to use P2 weighting models rather than SDEdit.**
 --->
### Set Up 
Create the environment by running the following:
```
- conda env create -f environment.yml
- pip install git+https://github.com/openai/CLIP.git
- pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl
```
Download the pretrained models from [here](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) and store it in `pretrained/`

You have to match `data.dataset` in `custom.yml` with your data domain. For example, if you want to use CelebaHQ, `data.dataset` should be `CelebA_HQ`. 

### Using Pretrained Models to generate debiased images
We provide pretrained H classifiers for  `Gender`, `Race` and `age` for Celeba-HQ images

Run the `./s_u_r.sh` file to generate images:

```
- `exp`: Path that the iamges should be stored in.
- `edit_attr`: Attribute to edit. But not used for now. you can use `./utils/text_dic.py` to predefined source-target text pairs or define new pair. 
- `n_test_img` : How many images should be generated?
- `attribute_list` : Attribute to be balanced: [1,0,0,0 - Eyeglasses, 0,1,0,0 - Gender, 0,0,1,0 - Race] [For multi attributes, add 1's accordingly, ex: 1,1,0,0 = Eyeglasses+ Gender]
- `scale` : Guidance scale [Hyperparameter] Refer to __ section for scale of the attributes in the paper, if not present, needs to be tuned
- `just_precompute` : False

Vanilla Generation:
- `vanilla_generation` : True

Sample based Generation:
- `vanilla_generation` : False
- `sample` : True
- `bs_test` : 1
- `male` : 0/1 [Binary, for each class of the attribute. Example: for gender, 0->Female, 1->Male]

Distribution based Generation:
- `vanilla_generation` : False
- `sample` : False
- `bs_test` : 100 (if any other, scale needs to be tuned accordingly) (higher the batch size, better is the guidance)
- `male` : Fraction of each class. example: for gender, 0.5 = 50% Male, 50% Female generations
```

### Training h-classifiers for other attributes/ other datasets
- A directory of images of all the classes of the particular attribute needs to be made. Example: If attribute is Gender, images of males and females need to be kept in 2 directories (Around 1000-2000 images for each class)
- Generated their h vectors :
    Run the  `script_extract_he.sh` file to extract the h vectors of images.

```
- `just_precompute`: True
- `test_path_one` : Path containing the images whose h-vectors should be generated.
- `savepath` : Path where the h vectors need to be saved.
```
- Train a Linear classifier on these images. Run __________ file to do this
- Use the checkpoints of the classifier for generating images. Copy the checkpoint path into `h_classification.py` (for sample based) and `multi_classifier.py` for distribution based, and use the respective index in the  `attribute_list` parameter while generating


## Acknowledge
[Codes are based on Asyrp.] (https://github.com/kwonminki/Asyrp_official)


## Bibtex
```
@article{parihar2024balancing,
  title={Balancing Act: Distribution-Guided Debiasing in Diffusion Models},
  author={Parihar, Rishubh and Bhat, Abhijnya and Mallick, Saswat and Basu, Abhipsa and Kundu, Jogendra Nath and Babu, R Venkatesh},
  journal={Computer Vision and Pattern Recognition},
  year={2024}
}
```
