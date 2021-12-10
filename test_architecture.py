import torch

import dnnlib
import legacy
from training.networks_encoder import Encoder


if __name__ == '__main__':
    # TODO: test for other configuration of encoder architectures
    device = torch.device('cuda:0')
    network_pkl = 'pretrained/stylegan3-t-ffhq-1024x1024.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    encoder = Encoder(pretrained=None).to(device)

    x = torch.randn((1,3,256,256)).to(device)
    print('input: [b,3,256,256]')
    print(x.shape)

    print('\nlatent: [b,16,512]')
    latent = encoder(x)
    print(latent.shape)

    synth = G.synthesis(latent)
    print('\nsynth: [b,3,1024,1024]')
    print(synth.shape)

    print("\nDone.")
