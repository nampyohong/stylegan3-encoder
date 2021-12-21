import torch

import dnnlib
import legacy
from training.networks_encoder import Encoder


def n_param(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad) 


if __name__ == '__main__':
    device = torch.device('cuda:0')
    network_pkl = 'pretrained/stylegan3-t-ffhq-1024x1024.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Base
#    encoder = Encoder(pretrained=None).to(device)
#
#    x = torch.randn((1,3,256,256)).to(device)
#    print('\nBase configuration')
#    print(f'\nencoder # params: {n_param(encoder)}')
#    print(f'encoder input_layer # params: {n_param(encoder.encoder.input_layer)}')
#    print(f'encoder body # params: {n_param(encoder.encoder.body)}')
#    print(f'encoder styles # params: {n_param(encoder.encoder.styles)}')
#    print(f'encoder latlayer # params: {n_param(encoder.encoder.latlayer1)+n_param(encoder.encoder.latlayer2)}')
#    print('\ninput: [b,3,256,256]')
#    print(x.shape)
#    print('latent: [b,16,512]')
#    latent = encoder(x)
#    print(latent.shape)
#    synth = G.synthesis(latent)
#    print('synth: [b,3,1024,1024]')
#    print(synth.shape)

    # Config-a
#    x = torch.randn((2,3,256,256)).to(device)
#    styleblock = dict(arch='transformer',  num_encoder_layers=1) # 1
#    encoder = Encoder(pretrained=None, **styleblock).to(device)
#    print('\nConfig-a: over parametrization')
#    print(f'\nencoder # params: {n_param(encoder)}')
#    print(f'encoder input_layer # params: {n_param(encoder.encoder.input_layer)}')
#    print(f'encoder body # params: {n_param(encoder.encoder.body)}')
#    print(f'encoder styles # params: {n_param(encoder.encoder.styles)}')
#    print(f'encoder latlayer # params: {n_param(encoder.encoder.latlayer1)+n_param(encoder.encoder.latlayer2)}')
#    print('\ninput: [b,3,256,256]')
#    print(x.shape)
#    print('latent: [b,16,512]')
#    latent = encoder(x)
#    print(latent.shape)
#    synth = G.synthesis(latent)
#    print('synth: [b,3,1024,1024]')
#    print(synth.shape)

    # Config-b train from w_avg
    w_avg = G.mapping.w_avg
    encoder = Encoder(pretrained=None,w_avg=w_avg).to(device)
    x = torch.randn((1,3,256,256)).to(device)
    print('\nBase configuration')
    print(f'\nencoder # params: {n_param(encoder)}')
    print(f'encoder input_layer # params: {n_param(encoder.encoder.input_layer)}')
    print(f'encoder body # params: {n_param(encoder.encoder.body)}')
    print(f'encoder styles # params: {n_param(encoder.encoder.styles)}')
    print(f'encoder latlayer # params: {n_param(encoder.encoder.latlayer1)+n_param(encoder.encoder.latlayer2)}')
    print('\ninput: [b,3,256,256]')
    print(x.shape)
    print('latent: [b,16,512]')
    latent = encoder(x)
    print(latent.shape)
    synth = G.synthesis(latent)
    print('synth: [b,3,1024,1024]')
    print(synth.shape)

    print("\nDone.")
