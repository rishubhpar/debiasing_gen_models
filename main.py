import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from diffusion_latent import Asyrp


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Logging
    # parser.add_argument('--sh_file_name', type=str, default='script.sh', help='copy the script this file')

    parser.add_argument('--lpips_edit_th', type=float, default=0.33, help='we use lpips_edit_th to get t_edit')
    parser.add_argument('--lpips_addnoise_th', type=float, default=1.2, help='we use lpips_addnoise_th to get t_addnoise')


    parser.add_argument('--just_precompute', action='store_true', help='just_precompute')

    # Training details
    # parser.add_argument('--use_x0_tensor', action='store_true', help='use_x0_tensor')
    # parser.add_argument('--save_x0', action='store_true', help='save x0_tensor (original image)')
    # parser.add_argument('--save_x_origin', action='store_true', help='save x_origin (original DDIM processing)')
    
    # parser.add_argument('--custom_train_dataset_dir', type=str, default="./custom/train")
    # parser.add_argument('--custom_test_dataset_dir', type=str, default="./custom/test")

    # Test Mode
    parser.add_argument('--run_test', action='store_true', help='run_test')
    # parser.add_argument('--load_random_noise', action='store_true', help='run_test')
    parser.add_argument('--saved_random_noise', action='store_true', help='run_test')

    
    parser.add_argument('--target_image_id', type=str, help='Sampling only one image which is target_image_id')
    parser.add_argument('--start_image_id', type=int, default=0, help='Sampling after start_image_id')
    
    parser.add_argument('--save_process_origin', action='store_true', help='save_origin_process')
    # parser.add_argument('--save_process_delta_h', action='store_true', help='save_delta_h_process')
    
    # parser.add_argument('--num_mean_of_delta_hs', type=int, default=0, help='Get mean of delta_h from num of data')

    # parser.add_argument('--do_alternate', type=int, default=0, help='Whether to train or not during CLIP finetuning')
    # parser.add_argument('--pass_editing', action='store_true', help='Whether to train or not during CLIP finetuning')
    
    # Style Transfer Mode
    parser.add_argument('--style_transfer', action="store_true")
    parser.add_argument('--style_transfer_style_from_train_images', default=False, action="store_true")
    parser.add_argument('--style_transfer_noise_from', type=str, default="contents")


    # LPIPS
    # parser.add_argument('--lpips', action="store_true")
    parser.add_argument('--custom_dataset_name', type=str, default="celeba")

    # Additional test
    parser.add_argument('--latent_classifier', action="store_true")
    parser.add_argument('--warigari', type=float, default=0.0)
    parser.add_argument('--attr_index', type=int)
    parser.add_argument('--classification_results_file_name', type=str, default="classification_results")
    parser.add_argument('--DirectionalClipSmilarity', action="store_true")

    # Mode
    parser.add_argument('--clip_finetune', action='store_true')
    parser.add_argument('--global_clip', action='store_true')
    parser.add_argument('--run_origin', action='store_true')
    parser.add_argument('--latent_at', action='store_true')
    parser.add_argument('--test_celeba_dialog', action='store_true')
    

    parser.add_argument('--save_to_folder', type=str)


    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')

    # Text
    # parser.add_argument('--edit_attr', type=str, default=None, help='Attribute to edit defiend in ./utils/text_dic.py')
    # parser.add_argument('--src_txts', type=str, action='append', help='Source text e.g. Face')
    # parser.add_argument('--trg_txts', type=str, action='append', help='Target text e.g. Angry Face')
    # parser.add_argument('--target_class_num', type=str, default=None)

    # Sampling
    parser.add_argument('--t_0', type=int, default=999, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=50, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=50, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=50, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')

    parser.add_argument('--LPIPS_addnoise_th', type=float, default=0.1, help='LPIPS_addnoise_th')


    # Train & Test
    parser.add_argument('--bs_test', type=int, default=1, help='Test batch size during CLIP fineuning')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    # parser.add_argument('--get_h_num', type=int, default=0, help='Training batch size during Latent CLR')
    

    parser.add_argument('--hs_coeff', type=float, default=0.9, help='hs coefficient')

    
    
    # Loss & Optimization
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16', help='ViT-B/16, ViT-B/32, RN50x16 etc')

    parser.add_argument('--savepath', type=str, default="./")
    parser.add_argument('--test_path_one', type=str, default="./")

    parser.add_argument('--sample', action="store_true")
    parser.add_argument('--male', type=float, default=1, help='Percentage of male')
    parser.add_argument('--eyeglasses', type=float, default=1, help='Percentage of eyeglasses')
    parser.add_argument('--scale', type=list_of_floats, default = [1500], help='Scaling the grad is multiplied with')
    parser.add_argument('--timestep_list', type=list_of_ints, default=[0,50], help='timesteps betweeen which scaling should be applied')
    parser.add_argument('--usefancy', action="store_true")
    parser.add_argument('--gamma_factor', type=float, default=0.1, help='Exponent in the gamma equation')
    parser.add_argument('--guidance_loss', type=str, default="chi_without_5", help="Choose one of 'kl_without_5, kl_with_5', 'chi_without_5', 'chi_with_5'")
    parser.add_argument('--attribute_list', type=list_of_ints, default=[0,1,0,0] , help='Eyeglasses,Gender,Race,Smile')
    parser.add_argument('--vanilla_generation', action="store_true")
    parser.add_argument('--universal_guidance', action="store_true")




    







    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    args.exp = args.exp + f'_LC_{new_config.data.category}_t{args.t_0}_ninv{args.n_inv_step}_ngen{args.n_train_step}'


    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # os.makedirs('checkpoint', exist_ok=True)
    # os.makedirs('checkpoint_latent', exist_ok=True)
    # os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs(args.exp, exist_ok=True)

    import shutil
    # if args.run_test:
    #     shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{(args.sh_file_name).split('.')[0]}_test.sh"))
    # elif args.style_transfer:
    #     shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{(args.sh_file_name).split('.')[0]}_style_transfer.sh"))
    # elif args.run_train:
    #     shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{args.sh_file_name.split('.')[0]}_train.sh"))
    # elif args.lpips:
    #     pass

    # args.training_image_folder = os.path.join(args.exp, 'training_images')
    # if not os.path.exists(args.training_image_folder):
    #     os.makedirs(args.training_image_folder)
    
    args.test_image_folder = os.path.join(args.exp, 'test_images', str(args.n_test_step))
    if not os.path.exists(args.test_image_folder):
        os.makedirs(args.test_image_folder)  

    # args.image_folder = os.path.join(args.exp, 'image_samples')
    # if not os.path.exists(args.image_folder):
    #     os.makedirs(args.image_folder)
    # else:
    #     overwrite = False
    #     if args.ni:
    #         overwrite = True
    #     else:
    #         response = input("Image folder already exists. Overwrite? (Y/N)")
    #         if response.upper() == 'Y':
    #             overwrite = True

    #     if overwrite:
    #         # shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder, exist_ok=True)
    #     else:
    #         print("Output image folder exists. Program halted.")
    #         sys.exit(0)

    if args.save_to_folder:
        args.training_image_folder = args.save_to_folder

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(args.timestep_list)
    print(args.guidance_loss)
    print(args.attribute_list)
    print("Vanilla??", args.vanilla_generation)

    print("seeeeeed", args.seed)
    

    # # This code is for me. If you don't need it, just remove it out.
    # if torch.cuda.is_available():
    #     assert args.bs_train % torch.cuda.device_count() == 0, f"Number of GPUs ({torch.cuda.device_count()}) must be a multiple of batch size ({args.bs_train})"

    runner = Asyrp(args, config) # if you want to specify the device, add device="something" in the argument
    try:
        # check the example script files for essential parameters
        # if args.run_train:
        #     runner.run_training()
        # elif args.run_test:
        runner.run_test()
        # elif args.lpips:
        #     runner.compute_lpips_distance()

    except Exception:
        logging.error(traceback.format_exc())

    return 0
          

if __name__ == '__main__':
    sys.exit(main())
