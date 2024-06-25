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

attribute_list="0,1,0,0"
scale=15
attribute=1
meh="vanilla"
# Next command
CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
                        --config $config                                    \
                        --exp ./runs/${attribute_list}/${meh}                               \
                        --n_test_img 5                                 \
                        --seed $RANDOM                                        \
                        --n_iter 5                                          \
                        --bs_test 1                                         \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step $test_step                            \
                        --lpips_addnoise_th 1.2                             \
                        --lpips_edit_th 0.33                                \
                        --savepath ""   \
                        --male $attribute      \
                        --timestep_list $timesteps    \
                        --scale $scale   \
                        --attribute_list $attribute_list    \
                        --sample 