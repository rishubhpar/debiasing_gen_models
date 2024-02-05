#!/bin/bash
# # For testing an already present attribute
# sh_file_name="s_u_r.sh"
# gpu="0"
# config="custom.yml"
# guid="black"

# test_step=50    # if large, it takes long time.
# dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.


# # RANDOM=$(date +%s)
# # RANDOM=$$
# # 20021
# timesteps="1,50"
# loss_type="chi_without_5"
# gamma=0.05

# attribute_list="0,1,0,0"
# scale=1500
# attribute=0.5
# # Next command
# CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
#                         --config $config                                    \
#                         --exp ./runs/rebuttal_distribution_more_flipped/${attribute_list}/$attribute                           \
#                         --edit_attr $guid                                   \
#                         --do_train 0                                      \
#                         --do_test 1                                         \
#                         --n_train_img 0                                   \
#                         --n_test_img 10000                                 \
#                         --seed 3090                                        \
#                         --n_iter 5                                          \
#                         --bs_train 1                                        \
#                         --bs_test 100                                         \
#                         --t_0 999                                           \
#                         --n_inv_step 50                                     \
#                         --n_train_step 50                                   \
#                         --n_test_step $test_step                            \
#                         --get_h_num 1                                       \
#                         --train_delta_block                                 \
#                         --sh_file_name $sh_file_name                        \
#                         --save_x_origin                                     \
#                         --use_x0_tensor                                     \
#                         --pass_editing                                      \
#                         --load_random_noise                                 \
#                         --dt_lambda $dt_lambda                              \
#                         --custom_train_dataset_dir "test_images/celeba/train"                \
#                         --custom_test_dataset_dir "/raid/rishubh/abhijnya/datasets/Celeba_HQ/smiling/"                  \
#                         --lpips_addnoise_th 1.2                             \
#                         --lpips_edit_th 0.33                                \
#                         --sh_file_name "s_u_r.sh"                \
#                         --savepath ""   \
#                         --male $attribute      \
#                         --timestep_list $timesteps    \
#                         --guidance_loss $loss_type  \
#                         --scale $scale   \
#                         --attribute_list $attribute_list    \
#                         --gamma_factor $gamma  \
#                         --usefancy  

start=`date +%s`


# For testing an already present attribute
sh_file_name="s_u_r.sh"
gpu="0"
config="custom.yml"
guid="black"

test_step=50    # if large, it takes long time.

# RANDOM=$(date +%s)
# RANDOM=$$
# 20021
timesteps="1,50"
loss_type="chi_without_5"
gamma=0.05

attribute_list="0,1,0,0"
scale=1500
attribute=0.5
meh="vanilla"
# Next command
CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
                        --config $config                                    \
                        --exp ./runs/delllll/${attribute_list}/${meh}                               \
                        --edit_attr $guid                                   \
                        --do_test 1                                         \
                        --n_train_img 0                                   \
                        --n_test_img 100                                 \
                        --seed $RANDOM                                        \
                        --n_iter 5                                          \
                        --bs_train 1                                        \
                        --bs_test 100                                         \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step $test_step                            \
                        --get_h_num 1                                       \
                        --train_delta_block                                 \
                        --sh_file_name $sh_file_name                        \
                        --save_x_origin                                     \
                        --use_x0_tensor                                     \
                        --pass_editing                                      \
                        --load_random_noise                                 \
                        --custom_train_dataset_dir "test_images/celeba/train"                \
                        --custom_test_dataset_dir "/raid/rishubh/abhijnya/datasets/Celeba_HQ/smiling/"                  \
                        --lpips_addnoise_th 1.2                             \
                        --lpips_edit_th 0.33                                \
                        --sh_file_name "s_u_r.sh"                \
                        --savepath ""   \
                        --male $attribute      \
                        --timestep_list $timesteps    \
                        --guidance_loss $loss_type  \
                        --scale $scale   \
                        --attribute_list $attribute_list    \
                        --gamma_factor $gamma  \
                        --usefancy  






# #!/bin/bash
# # For testing an already present attribute
# sh_file_name="s_u_r.sh"
# gpu="1"
# config="custom.yml"
# guid="black"

# test_step=50    # if large, it takes long time.
# dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.


# # RANDOM=$(date +%s)
# # RANDOM=$$
# # 20021
# timesteps="1,50"
# loss_type="chi_without_5"
# gamma=0.05

# attribute_list="0,1,0,0"
# scale=15
# attribute=0
# meh="vanilla"
# # Next command
# CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
#                         --config $config                                    \
#                         --exp ./runs/rebuttal/${attribute_list}/${meh}                                \
#                         --edit_attr $guid                                   \
#                         --do_train 0                                      \
#                         --do_test 1                                         \
#                         --n_train_img 0                                   \
#                         --n_test_img 4000                                 \
#                         --seed 3090                                        \
#                         --n_iter 5                                          \
#                         --bs_train 1                                        \
#                         --bs_test 1                                         \
#                         --t_0 999                                           \
#                         --n_inv_step 50                                     \
#                         --n_train_step 50                                   \
#                         --n_test_step $test_step                            \
#                         --get_h_num 1                                       \
#                         --train_delta_block                                 \
#                         --sh_file_name $sh_file_name                        \
#                         --save_x_origin                                     \
#                         --use_x0_tensor                                     \
#                         --pass_editing                                      \
#                         --load_random_noise                                 \
#                         --dt_lambda $dt_lambda                              \
#                         --custom_train_dataset_dir "test_images/celeba/train"                \
#                         --custom_test_dataset_dir "/raid/rishubh/abhijnya/datasets/Celeba_HQ/smiling/"                  \
#                         --lpips_addnoise_th 1.2                             \
#                         --lpips_edit_th 0.33                                \
#                         --sh_file_name "s_u_r.sh"                \
#                         --savepath ""   \
#                         --male $attribute      \
#                         --timestep_list $timesteps    \
#                         --guidance_loss $loss_type  \
#                         --scale $scale   \
#                         --attribute_list $attribute_list    \
#                         --gamma_factor $gamma  \
#                         --usefancy  \
#                         --vanilla_generation