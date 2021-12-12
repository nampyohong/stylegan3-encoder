import copy
import json
import os
import random
import pickle
import time
from pprint import pprint

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
from training.ranger import Ranger

#----------------------------------------------------------------------------

def save_image(images, save_path, gh, gw, H, W):
    np_imgs = []
    for i, image in enumerate(images):
        image = images[i][None,:,:]
        image = (image.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8).cpu().numpy()
        np_imgs.append(np.asarray(PIL.Image.fromarray(image[0], 'RGB').resize((H,W),PIL.Image.LANCZOS)))
    np_imgs = np.stack(np_imgs)
    np_imgs = np_imgs.reshape(gh,gw,H,W,3)
    np_imgs = np_imgs.transpose(0,2,1,3,4)
    np_imgs = np_imgs.reshape(gh*H, gw*W, 3)
    PIL.Image.fromarray(np_imgs, 'RGB').save(save_path)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    rank                    = 0,        # Rank of the current process in [0, num_gpus].
    num_gpus                = 1,        # Number of GPUs participating in the training.
    learning_rate           = 0.001,    # Learning rate
    lambda1                 = 1.0,      # L2 loss multiplier factor
    lambda2                 = 0.8,      # LPIPS loss multiplier factor
    lambda3                 = 0.1,      # ID loss multiplier factor
    batch_size              = 32,       # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    random_seed             = 0,        # Global random seed.
    num_workers             = 3,        # Dataloader workers.
    resume_pkl              = None,     # Network pickle to resume training from.
    training_steps          = 500000,   # Total training batch steps
    val_steps               = 1000,     # Validation batch steps 
    print_steps             = 50,       # How often to print logs
    tensorboard_steps       = 50,       # How often to log to tensorboard?
    image_snapshot_steps    = 100,       # How often to save image snapshots? None=disable.
    network_snapshot_steps  = 1000,     # How often to save network snapshots?
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
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
    # TODO : set dataset_type from config
    dataset_type = 'ffhq' # ffhqs
    dataset_dir = f'data/{dataset_type}'
    training_set = ImagesDataset(dataset_dir, mode='train')
    training_set_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=num_gpus, rank=rank, shuffle=True, seed=random_seed, drop_last=False)
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

    # Initialize logs.
    if rank == 0:
        print('Initialize tensorboard logs...')
        logger = tensorboard.SummaryWriter(run_dir)

    # Train.
    E.train()
    G.eval()

    #TODO : implement validation
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
            loss_dict = {} # for tb logs
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

            if rank == 0 and cur_step % print_steps == 0:
                print(f'\nCurrent batch step: {cur_step}')
                pprint(loss_dict)

            # back propagation
            loss.backward()

            # optimizer step
            optimizer.step()

            # barrier
            torch.distributed.barrier()

            # Save image snapshot.
            if rank == 0 and cur_step % image_snapshot_steps == 0:
                print(f"Saving image snapshot at step {cur_step}...")
                gh, gw = 1, batch_gpu
                H,W = real_images.shape[2], real_images.shape[3]
                real_path = f'image-snapshot-real-{cur_step:06d}.png'
                generated_path = f'image-snapshot-generated-{cur_step:06d}.png'
                save_image(real_images, os.path.join(run_dir, real_path), gh, gw, H, W)
                save_image(generated_images, os.path.join(run_dir, generated_path), gh, gw, H, W)

            # Save network snapshot.
            network_pkl = None
            snapshot_data = None
            if rank == 0 and cur_step % network_snapshot_steps == 0:
                print(f"Saving netowrk snapshot at step {cur_step}...")
                snapshot_data = dict(E=E, G=G)
                for key, value in snapshot_data.items():
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
#                        if num_gpus > 1:
#                            misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
#                            for param in misc.params_and_buffers(value):
#                                torch.distributed.broadcast(param, src=0)
                        snapshot_data[key] = value.cpu()
                    del value # conserve memory
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_step:06d}.pkl')
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
            del snapshot_data # conserve memory

            # Tensorboard logs.
            if rank == 0 and cur_step % tensorboard_steps == 0:
                for key in loss_dict:
                    logger.add_scalar(f'train/{key}', loss_dict[key], cur_step)
            # barrier
            torch.distributed.barrier()

            # update cur_step
            cur_step += 1
            if cur_step == training_steps:
                break

    # Done.
    torch.distributed.destroy_process_group()

    if rank == 0:
        print()
        print('Exiting...')
