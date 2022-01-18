import copy
import json
import os
import random
import pickle
import time
from pprint import pprint
from tqdm import tqdm

import PIL.Image
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
from lpips import LPIPS
from torch.nn.parallel import DistributedDataParallel as DDP

import dnnlib
import legacy
from torch_utils import misc
from training.dataset_encoder import ImagesDataset
from training.loss_encoder import l2_loss, IDLoss
from training.networks_encoder import Encoder
from training.training_loop_encoder import save_image

#----------------------------------------------------------------------------

@torch.no_grad()
def testing_loop(
    test_dir                = '.',          # Output directory.
    rank                    = 0,            # Rank of the current process in [0, num_gpus].
    model_architecture      = 'base',       # Model architecture type, ['base', 'transformer']
    w_avg                   = False,        # Train delta w from w_avg
    num_encoder_layers      = 1,            # Encoder layers if model_architecture is transformer
    dataset_dir             = 'celeba-hq',  # Train dataset directory
    num_gpus                = 8,            # Number of GPUs participating in the training.
    batch_size              = 32,           # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,            # Number of samples processed at a time by one GPU.
    generator_pkl           = None,         # Generator pickle to encode.
    l2_lambda               = 1.0,          # L2 loss multiplier factor
    lpips_lambda            = 0.8,          # LPIPS loss multiplier factor
    id_lambda               = 0.1,          # ID loss multiplier factor
    reg_lambda              = 0.0,          # e4e reg loss multiplier factor
    gan_lambda              = 0.0,          # e4e latent gan loss multiplier factor
    edit_lambda             = 0.0,          # e4e editability loss multiplier factor
    random_seed             = 0,            # Global random seed.
    num_workers             = 3,            # Dataloader workers.
    cudnn_benchmark         = True,         # Enable torch.backends.cudnn.benchmark?
):

    # initialize
    device = torch.device('cuda', rank)

    # Reproducability
    random.seed(random_seed * num_gpus + rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.cuda.manual_seed(random_seed * num_gpus + rank)
    torch.cuda.manual_seed_all(random_seed * num_gpus + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark 
    
    # Load testing set.
    if rank == 0:
        print('Loading testing set...')
    testing_set = ImagesDataset(dataset_dir, mode='test')
    testing_set_sampler = torch.utils.data.distributed.DistributedSampler(testing_set, num_replicas=num_gpus, rank=rank, shuffle=False, seed=random_seed, drop_last=False)
    testing_loader = torch.utils.data.DataLoader(dataset=testing_set, sampler=testing_set_sampler, batch_size=batch_size//num_gpus, num_workers=num_workers)
    if rank == 0:
        print()
        print('Num images: ', len(testing_set))
        print('Image shape:', testing_set.__getitem__(0)[0].shape)
        print()

    # Construct generator.
    if rank == 0:
        print('Constructing generator...')
    with dnnlib.util.open_url(generator_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Initizlize loss
    if rank == 0:
        print('Initialize loss...')
    id_loss = IDLoss().to(device)
    lpips_loss = LPIPS(net='alex', verbose=False).to(device).eval()

    # Initialize logs.
    if rank == 0:
        print('Initialize tensorboard logs...')
        logger = tensorboard.SummaryWriter(test_dir)

    # Test.
    G.eval()

    latent_avg = None
    if w_avg:
        latent_avg = G.mapping.w_avg

    test_pkl_lst = [os.path.join(test_dir, 'network_snapshots', x) for x in sorted(os.listdir(os.path.join(test_dir, 'network_snapshots')))][-9:]

    for test_pkl in test_pkl_lst:
        if rank == 0:
            print(f'\nConstructing encoder from: {test_pkl}')
        # Construct encoder.
        if model_architecture == 'base':
            E = DDP(Encoder(pretrained=test_pkl,w_avg=latent_avg).to(device), device_ids=[rank])
        elif model_architecture == 'transformer':
            styleblock = dict(arch='transformer', num_encoder_layers=num_encoder_layers)
            E = DDP(Encoder(pretrained=test_pkl, w_avg=latent_avg, **styleblock).to(device), device_ids=[rank])
        cur_step = E.module.resume_step
        assert cur_step.__repr__() in test_pkl

        E.eval()

        epoch_loss = 0.0
        epoch_loss_dict = {k:0.0 for k in ['l2', 'lpips', 'id', 'id_improve', 'loss']}

        for batch_idx, batch in tqdm(enumerate(testing_loader),total=len(testing_loader)):
            # x:source image = y:real image
            # E(x): w, encoded latent
            # G.synthesis(E(x)):encoded_images
            x,y = batch 
            x,real_images = x.to(device),y.to(device)
            face_pool=torch.nn.AdaptiveAvgPool2d((256,256))
            encoded_images = face_pool(G.synthesis(E(x)))
            
            # get loss
            loss = 0.0
            loss_dict = {}
            loss_l2 = l2_loss(encoded_images, real_images)
            loss_dict['l2'] = loss_l2.item()
            loss += loss_l2 * l2_lambda
            loss_lpips = lpips_loss(encoded_images, real_images).squeeze().mean()
            loss_dict['lpips'] = loss_lpips.item()
            loss += loss_lpips * lpips_lambda
            loss_id, sim_improvement = id_loss(encoded_images, real_images, x)
            loss_dict['id'] = loss_id.item()
            loss_dict['id_improve'] = sim_improvement
            loss += loss_id * id_lambda
            loss_dict['loss'] = loss.item()
            
            epoch_loss += loss.item()
            for k in epoch_loss_dict:
                epoch_loss_dict[k] += loss_dict[k]

            # barrier
            torch.distributed.barrier()
            torch.cuda.empty_cache()
    
            # Save image snapshot.
            if rank == 0 and batch_idx == 0:
                print(f"Saving image samples...")
                gh, gw = 1, batch_gpu
                H,W = real_images.shape[2], real_images.shape[3]
                real_path = f'test-image-snapshot-real-{cur_step:06d}.png'
                encoded_path = f'test-image-snapshot-encoded-{cur_step:06d}.png'
                save_image(real_images, os.path.join(test_dir, 'image_snapshots', real_path), gh, gw, H, W)
                save_image(encoded_images, os.path.join(test_dir, 'image_snapshots', encoded_path), gh, gw, H, W)
    
            # barrier
            torch.distributed.barrier()

        # Tensorboard logs.
        # TODO: need to get other devices' loss
        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= len(testing_loader)
        if rank == 0:
            pprint(epoch_loss_dict)
            for key in epoch_loss_dict:
                logger.add_scalar(f'test/{key}', epoch_loss_dict[key], cur_step)
        # barrier
        torch.distributed.barrier()

        del E

    # Done.
    torch.distributed.destroy_process_group()

    if rank == 0:
        print()
        print('Exiting...')
