# Balancing Act: Distribution-Guided Debiasing in Diffusion Models, CVPR 2024 

[![arXiv](https://img.shields.io/badge/arxiv-2309.05569-red)](https://arxiv.org/abs/2402.18206)
[![Webpage](https://img.shields.io/badge/Webpage-green)](https://ab-34.github.io/balancing_act/)
[![slides](https://img.shields.io/badge/Slides-orange)](https://docs.google.com/presentation/d/1mQOl3KH9ddcouBA11-VarATyyhAiGl8nK94ot0aQe14/edit?usp=sharing)

### Set Up 
Create the environment by running the following:
```bash
conda env create -f environment.yml
```
Download the pretrained models from [here](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) and store it in `pretrained/`

You have to match `data.dataset` in `custom.yml` with your data domain. For example, if you want to use CelebaHQ, `data.dataset` should be `CelebA_HQ`. 

### Using Pretrained Models to generate debiased images
We provide pretrained H classifiers for  `Gender`, `Race` and `age` for Celeba-HQ images

Run the `./s_u_r.sh` file to generate images:

<details>
<summary><span style="font-weight: bold;">Necessary arguments for generation</span></summary>

- `exp`: Path that the images should be stored in.
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
</details>

### Training h-classifiers for other attributes/ other datasets
- A directory of images of all the classes of the particular attribute needs to be made. Example: If attribute is Gender, images of males and females need to be kept in 2 directories (Around 1000-2000 images for each class)
- Run the  `script_extract_he.sh` file to extract the h vectors of images.

<details>
<summary><span style="font-weight: bold;">Arguments for training</span></summary>

- `just_precompute`: True
- `test_path_one` : Path containing the images whose h-vectors should be generated.
- `savepath` : Path where the h vectors need to be saved.
</details>

- Train a Linear classifier on these images. Run __________ file to do this
- Use the checkpoints of the classifier for generating images. Copy the checkpoint path into `h_classification.py` (for sample based) and `multi_classifier.py` for distribution based, and use the respective index in the  `attribute_list` parameter while generating.


## Stable Diffusion Implementation
We use a modded version of [HF diffusers](https://github.com/huggingface/diffusers) to introduce guidance in the hvectors. The loss strengths have been adjusted for generating a batch of 4 images on a 12GB GPU. Some adjustments to this value will be needed to achieve proper debiasing on larger batches.

```bash
pip install huggingface_hub transformers accelerate
python run_stable_diffusion.py
```

<details>
<summary><span style="font-weight: bold;">Arguments for generating from stable diffusion</span></summary>

- `original_prompt`: The prompt of the subject that needs to be generated. Ex: person, doctor, constsruction worker, firefighter, etc.
- `negative_prompt`: We inherit this from a [discussion thread in HF](https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7857#63bee17e20784381e8e54d33) and found it suitable for humans and most subjects covered in our paper.
- `MODE`: We offer two solutions: `sampled` and `distribution`. The former allows us to generate samples solely for a particular biased class (e.g., all males), while the latter generates a balanced distribution.
- `checkpoint_path`: The path to the pretrained classifiers.
- `loss_strength`: The multiplier to the guidance strength of the classifier for strong effects.
- `scaling_strength`: The multiplier to the guidance strength of the classifier for milder effects.

</details>

## Acknowledge
Codes are based on [Asyrp](https://github.com/kwonminki/Asyrp_official)


## Bibtex
```
@InProceedings{Parihar_2024_CVPR,
    author    = {Parihar, Rishubh and Bhat, Abhijnya and Basu, Abhipsa and Mallick, Saswat and Kundu, Jogendra Nath and Babu, R. Venkatesh},
    title     = {Balancing Act: Distribution-Guided Debiasing in Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6668-6678}
}
```
