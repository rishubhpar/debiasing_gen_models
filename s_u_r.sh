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
                        --n_test_img 10                                 \
                        --seed $RANDOM                                        \
                        --n_iter 5                                          \
                        --bs_test 100                                         \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step $test_step                            \
                        --lpips_addnoise_th 1.2                             \
                        --lpips_edit_th 0.33                                \
                        --savepath ""   \
                        --male $attribute      \
                        --timestep_list $timesteps    \
                        --guidance_loss $loss_type  \
                        --scale $scale   \
                        --attribute_list $attribute_list    \
                        --gamma_factor $gamma  \
                        --usefancy  \
                        --vanilla_generation