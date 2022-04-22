import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from model.cd_modules.cd_head import cd_head 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_false')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            train_set   = Data.create_cd_dataset(dataset_opt, phase)
            train_loader= Data.create_dataloader(
                train_set, dataset_opt, phase)
            val_set   = Data.create_cd_dataset(dataset_opt, 'val')
            val_loader= Data.create_dataloader(
                train_set, dataset_opt, 'val')
        elif phase == 'val':
            print("Unconditional Sampling. No validation dataloader required.")
            # val_set = Data.create_dataset(dataset_opt, phase)
            # val_loader = Data.create_dataloader(
            # val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')


    # diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    #Set noise schedule
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # CD model
    change_detection = Model.create_CD_model(opt)
    
    # Training loop
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']
    
    # Directory to save feature maps
    feature_path = '{}/features'.format(opt['path']['results'])
    os.makedirs(feature_path, exist_ok=True)
                
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                
                # Feeding data to diffusion model
                diffusion.feed_data(train_data)
                feats_A, feats_B = diffusion.get_feats(t=2)

                # Feeding features from diffusion model to train the cd model
                change_detection.feed_data(feats_A, feats_B, train_data)
                change_detection.optimize_parameters()

                # Obtaining features of pre-change and post-change images through DDPM
                # t=0
                # level=0
                
                # img_A = Metrics.tensor2img(train_data["A"][0,:,:,:])  # uint8
                # Metrics.save_img(img_A, '{}/img_A.png'.format(feature_path))

                # img_B = Metrics.tensor2img(train_data["B"][0,:,:,:])  # uint8
                # Metrics.save_img(img_B, '{}/img_B.png'.format(feature_path))

                # for i in range(feats_A[level].size(1)):
                #     feat_img_A = Metrics.tensor2img(feats_A[level][0,i,:,:])  # uint8
                #     Metrics.save_feat(feat_img_A, '{}/feat_A_{}_level_{}_t_{}.png'.format(feature_path, i, level, t))

                #     feat_img_B = Metrics.tensor2img(feats_B[level][0,i,:,:])  # uint8
                #     Metrics.save_feat(feat_img_B, '{}/feat_B_{}_level_{}_t_{}.png'.format(feature_path, i, level, t))
                

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = change_detection.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                
                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    for idx in range(0, opt['datasets']['val']['data_len'], 1):
                        change_detection.test()
                        visuals = change_detection.get_current_visuals()

                        img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                        # generation
                        Metrics.save_img(
                            img_A, '{}/img_A_{}_{}.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            img_B, '{}/img_B_{}_{}.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            pred_cm, '{}/pred_{}_{}.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            gt_cm, '{}/gt_{}_{}.png'.format(result_path, current_step, idx))
                        
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (img_A, img_B, pred_cm, gt_cm), axis=1), [2, 0, 1]),
                            idx)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate(sam_img)
                            )

                    # log
                    #logger_val = logging.getLogger('val')  # validation logger
                    #logger_val.info('<epoch:{:3d}, iter:{:8,d}> Sample generation completed.'.format(
                        #current_epoch, current_step))

                    if wandb_logger:
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    change_detection.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for idx in range(0, opt['datasets']['val']['data_len']):
            diffusion.test(in_channels=opt['model']['unet']['in_channel'], img_size=opt['datasets']['val']['resolution'], continous=True)
            visuals = diffusion.get_current_visuals()

            img_mode = 'grid'
            if img_mode == 'single':

                # single img series
                sam_img = visuals['SAM']
                sample_num = sam_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sam_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sam_img = Metrics.tensor2img(visuals['SAM'])  # uint8
                Metrics.save_img(
                    sam_img, '{}/sampling_process_{}_{}.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SAM'][-1]), '{}/sample_{}_{}.png'.format(result_path, current_step, idx))

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(sam=Metrics.tensor2img(visuals['SAM'][-1]))

        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> Sample generation completed.'.format(
            current_epoch, current_step))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
