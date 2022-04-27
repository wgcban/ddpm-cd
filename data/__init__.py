'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

#Create CD dataloader
def create_cd_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train' or 'val' or 'test':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

# Create image dataset
def create_image_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.ImageDataset import ImageDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                resolution=dataset_opt['resolution'],
                split=phase,
                data_len=dataset_opt['data_len']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

# Create change-detection dataset
def create_cd_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.CDDataset import CDDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                resolution=dataset_opt['resolution'],
                split=phase,
                data_len=dataset_opt['data_len']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name'],
                                                           phase))
    return dataset