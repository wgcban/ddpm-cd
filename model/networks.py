import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
from model.cd_modules.cd_head_v2 import cd_head_v2, get_in_channels
from thop import profile, clever_format
import copy
import time
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type=model_opt['diffusion']['loss'],    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        print("Distributed training")
        netG = nn.DataParallel(netG)
    return netG


# Change Detection Network
def define_CD(opt):
    cd_model_opt = opt['model_cd']
    diffusion_model_opt = opt['model']
    
    # Define change detection network head
    netCD = cd_head_v2(feat_scales=cd_model_opt['feat_scales'],
                       out_channels=cd_model_opt['out_channels'],
                       inner_channel=diffusion_model_opt['unet']['inner_channel'],
                       channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
                       img_size=cd_model_opt['output_cm_size'],
                       time_steps=cd_model_opt["t"])
    
    # Initialize the change detection head if it is 'train' phase 
    if opt['phase'] == 'train':
        # Try different initialization methods
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netCD, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netCD = nn.DataParallel(netCD)
    
    ### Profiling ###
    f_A, f_B = [], [] 
    feat_scales = cd_model_opt['feat_scales'].copy()
    feat_scales.sort(reverse=True)
    h, w = 8, 8
    for i in range(0, len(feat_scales)):
        dim = get_in_channels([feat_scales[i]], diffusion_model_opt['unet']['inner_channel'], diffusion_model_opt['unet']['channel_multiplier'])
        A = torch.randn(1, dim, h, w).cuda()
        B = torch.randn(1, dim, h, w).cuda()
        f_A.append(A)
        f_B.append(B)
        f_A.append(A)
        f_B.append(B)
        f_A.append(A)
        f_B.append(B)
        h *= 2
        w *= 2
    f_A_r = [ele for ele in reversed(f_A)]
    f_B_r = [ele for ele in reversed(f_B)]

    F_A=[]
    F_B=[]
    for t_i in range(0, len(cd_model_opt["t"])):
        print(t_i)
        F_A.append(f_A_r)
        F_B.append(f_B_r)
    flops, params = profile(copy.deepcopy(netCD).cuda(), inputs=(F_A, F_B,), verbose=False)
    flops, params = clever_format([flops, params])
    netGcopy = copy.deepcopy(netCD).cuda()
    netGcopy.eval()
    with torch.no_grad():
        start = time.time()
        _ = netGcopy(F_A, F_B)
        end = time.time()
    print('### Model Params: {} FLOPs: {} Time: {}ms ####'.format(params, flops, 1000*(end-start)))
    del netGcopy, F_A, F_B, f_A_r, f_B_r, f_A, f_B
    ### --- ###
    return netCD
