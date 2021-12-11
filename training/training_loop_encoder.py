import copy
import json
import os
import random
import pickle
import time

import PIL.Image
import numpy as np
import torch
from lpips import LPIPS
from torch.nn.parallel import DistributedDataParallel as DDP

import dnnlib
import legacy
from torch_utils import misc
from training.dataset_encoder import ImagesDataset
from training.loss_encoder import l2_loss, IDLoss
from training.networks_encoder import Encoder
from training.ranger import Ranger

#----------------------------------------------------------------------------

def training_loop(
    rank            = 0,        # Rank of the current process in [0, num_gpus].
    num_gpus        = 1,        # Number of GPUs participating in the training.
    learning_rate   = 0.001,    # Learning rate
    batch_size      = 32,       # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu       = 4,        # Number of samples processed at a time by one GPU.
    random_seed     = 0,        # Global random seed.
    num_workers     = 3,        # Dataloader workers.
    resume_pkl      = None,     # Network pickle to resume training from.
    cudnn_benchmark = False,    # Enable torch.backends.cudnn.benchmark?
    training_steps  = 500000,   # Total training batch steps
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
    
    # Load training set.
    if rank == 0:
        print('Loading training set...')
    dataset_dir = 'data/ffhqs'
    training_set = ImagesDataset(dataset_dir, mode='train')
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_loader = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, num_workers=num_workers)
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.__getitem__(0)[0].shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    network_pkl = 'pretrained/stylegan3-t-ffhq-1024x1024.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    E = DDP(Encoder(pretrained=None).to(device), device_ids=[rank])

    # TODO: Resume from existing pickle.
    cur_step = 0

    # Initizlize loss
    if rank == 0:
        print('Initialize loss...')
    id_loss = IDLoss().to(device)
    lpips_loss = LPIPS(net='alex', verbose=False).to(device).eval()
    lambda1, lambda2, lambda3 = 1.0,0.8,0.1

    # Initialize optimizer
    if rank == 0: 
        print('Initialize optimizer...')
    params = list(E.parameters())
    optimizer = Ranger(params, lr=learning_rate)

    # train
    E.train()
    G.eval()

    training_steps = 10 # FIXME for test
    while cur_step < training_steps:
        for batch_idx, batch in enumerate(training_loader):
            optimizer.zero_grad()
            # x:source image = y:real image
            # E(x): w, encoded latent
            # G(E(x)):generated_images
            x,y = batch 
            x,real_images = x.to(device),y.to(device)
            w = E(x)
            face_pool=torch.nn.AdaptiveAvgPool2d((256,256))
            generated_images = face_pool(G.synthesis(E(x)))
            
            # get loss
            loss = 0.0
            loss_dict = {}

            loss_l2 = l2_loss(generated_images, real_images)
            loss_dict['l2'] = loss_l2.item()
            loss += loss_l2 * lambda1

            loss_lpips = lpips_loss(generated_images, real_images).squeeze().mean()
            loss_dict['lpips'] = loss_lpips.item()
            loss += loss_lpips * lambda2

            loss_id, sim_improvement = id_loss(generated_images, real_images, x)
            loss_dict['id'] = loss_id.item()
            loss_dict['id_improve'] = sim_improvement
            loss += loss_id * lambda3

            # back propagation
            loss.backward()
            # optimizer step
            optimizer.step()
            cur_step += 1
            if cur_step == training_steps:
                break

    # Save image snapshot.
    # Save network snapshot.
    # Done.
    torch.distributed.destroy_process_group()

    if rank == 0:
        print()
        print('Exiting...')
