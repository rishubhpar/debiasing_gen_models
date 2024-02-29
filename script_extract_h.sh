# For testing an already present attribute
sh_file_name="s_u_r.sh"
gpu="0"
config="custom.yml"
guid="black"
test_step=50    # if large, it takes long time.

# Next command
CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
                        --config $config                                    \
                        --exp ./runs/${guid}                                \
                        --edit_attr $guid                                   \
                        --do_test 1                                         \
                        --n_train_img 0                                   \
                        --n_test_img 10                                    \
                        --n_iter 5                                          \
                        --bs_train 1                                        \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step $test_step                            \
                        --get_h_num 1                                       \
                        --train_delta_block                                 \
                        --sh_file_name $sh_file_name                        \
                        --save_x0                                           \
                        --use_x0_tensor                                     \
                        --custom_train_dataset_dir "test_images/celeba/train"                \
                        --custom_test_dataset_dir "/raid/rishubh/abhijnya/datasets/Celeba_HQ/smiling/"                  \
                        --lpips_addnoise_th 1.2                             \
                        --lpips_edit_th 0.33                                \
                        --sh_file_name "script_extract_h.sh"                \
                        --just_precompute                                       \
                        --savepath "/data/abhijnya/Classifier_guidance/runs"   \
                        --test_path_one  "/data/abhijnya/Asyrp_official/input/dclf/new_model/gender/male"