from pprint import pprint

import PIL.Image
import numpy as np
import torch

import dnnlib
import legacy
from training.loss_encoder import l2_loss, IDLoss
from lpips import LPIPS


if __name__ == '__main__':
    device = torch.device('cuda:0')

    # generate 2 images
    network_pkl = 'pretrained/stylegan3-t-ffhq-1024x1024.pkl'
    truncation_psi = 0.5
    noise_mode = 'const'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    seeds = [0,1]
    imgs = []
    for seed_idx, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1,G.z_dim)).to(device)
        img = G(z,label,truncation_psi=truncation_psi,noise_mode=noise_mode)
        imgs.append(img)
    generated_images, real_images = imgs[0], imgs[1]

    print(f'\ngenerated image shape : {imgs[0].shape}') # imgs[0]
    print(f'train image shape : {imgs[1].shape}') # imgs[1]
    print()

    # get loss
    id_loss = IDLoss().to(device)
    lpips_loss = LPIPS(net='alex').to(device).eval()
    loss_dict = dict()
    loss = 0.0
    # lambda1 : pixelwise l2 loss
    # lambda2 : lpips loss
    # lambda3 : id_loss
    # TODO : get lambda values from config
    lambda1, lambda2, lambda3 = 1,0.8,0.1
    loss_l2 = l2_loss(generated_images, real_images)
    loss_dict['l2'] = loss_l2.item()
    loss += loss_l2 * lambda1

    loss_lpips = lpips_loss(generated_images, real_images).squeeze()
    loss_dict['lpips'] = loss_lpips.item()
    loss += loss_lpips * lambda2
    
    loss_id, sim_improvement = id_loss(generated_images, real_images, real_images)
    loss_dict['id'] = loss_id.item()
    loss_dict['id_improve'] = sim_improvement
    loss += loss_id * lambda3
    
    print(f'\nloss: {loss}')
    print('\nloss dictionary')
    pprint(loss_dict)
    print('Done.')
