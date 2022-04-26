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

    # Loading change-detction datasets.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating [train] change-detection dataloader.")
            train_set   = Data.create_cd_dataset(dataset_opt, phase)
            train_loader= Data.create_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val':
            print("Creating [val] change-detection dataloader.")
            val_set   = Data.create_cd_dataset(dataset_opt, phase)
            val_loader= Data.create_cd_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # Creating change-detection model
    change_detection = Model.create_CD_model(opt)
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):         
            change_detection._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                 ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            
            ################
            ### training ###
            ################
            for current_step, train_data in enumerate(train_loader):
                # Feeding data to diffusion model and get features
                diffusion.feed_data(train_data)
                feats_A, feats_B = diffusion.get_feats(t=2)

                # Feeding features from the diffusion model to the CD model
                change_detection.feed_data(feats_A, feats_B, train_data)
                change_detection.optimize_parameters()
                change_detection._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = change_detection.get_current_log()
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                    logger.info(message)

                    #vissuals
                    visuals = change_detection.get_current_visuals()

                    # Converting to uint8
                    img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                    img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                    gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                    pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                    #save imgs
                    Metrics.save_img(
                        img_A, '{}/img_A_b{}.png'.format(train_result_path, current_step))
                    Metrics.save_img(
                        img_B, '{}/img_B_b{}.png'.format(train_result_path, current_step))
                    Metrics.save_img(
                        pred_cm, '{}/pred_b{}.png'.format(train_result_path, current_step))
                    Metrics.save_img(
                        gt_cm, '{}/gt_b{}.png'.format(train_result_path, current_step))


                    # Uncommet the following line to visualize features from the diffusion model
                    # print_feats(opt, train_data, level=3)
                
            ### log epoch status ###
            change_detection._collect_epoch_states()
            logs = change_detection.get_current_log()
            message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)

            if wandb_logger:
                wandb_logger.log_metrics(logs)

            change_detection._clear_cache()
            
            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    diffusion.feed_data(val_data)
                    feats_A, feats_B = diffusion.get_feats(t=2)

                    # Feed data to CD model
                    change_detection.feed_data(feats_A, feats_B, val_data)
                    change_detection.test()
                    change_detection._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = change_detection.get_current_log()
                        logger_val  = logging.getLogger('val')  # validation logger
                        message     = '[Validation CD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['running_acc'])
                        logger_val.info(message)

                        #vissuals
                        visuals = change_detection.get_current_visuals()

                        # Converting to uint8
                        img_A   = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                        #save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                        
                        tb_logger.add_image(
                            'Iter_{}'.format(current_epoch),
                            np.transpose(np.concatenate((img_A, img_B, pred_cm, gt_cm), axis=1), [2, 0, 1]), 
                            current_step)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{current_step}', 
                                    np.concatenate((img_A, img_B, pred_cm, gt_cm), axis=1)
                                )

                change_detection._collect_epoch_states()
                logs     = change_detection.get_current_log()
                message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/mF1': logs['epoch_acc'],
                        'validation/val_epoch': current_epoch
                    })
                
                if logs['epoch_acc'] > best_mF1:
                    is_best_model = True
                    best_mF1 = logs['epoch_acc']
                    logger.info('Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('Saving the current cd model and training states.')

                change_detection.save_network(current_epoch, is_best_model = is_best_model)

                if wandb_logger and opt['log_wandb_ckpt']:
                    wandb_logger.log_checkpoint(current_epoch, 0)
                
                change_detection._clear_cache()

                val_step += 1

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
