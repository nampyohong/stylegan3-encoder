import pickle

import torch

import dnnlib
import legacy
from training.dataset_encoder import ImagesDataset
from training.networks_encoder import Encoder
from training.training_loop_encoder import save_image


if __name__ == '__main__':
    device = torch.device('cuda:0')
    with dnnlib.util.open_url('pretrained/stylegan3-t-ffhq-1024x1024.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    pretrained = 'exp/base/00000-base-ffhq-gpus8-batch32/network_snapshots/network-snapshot-100000.pkl'
    E = Encoder(pretrained=pretrained).to(device)

    test_set = ImagesDataset('data/celeba-hq-samples', mode='inference')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=len(test_set))
    print(f'\ninference dataset length: {len(test_set)}')

    print(f'\ninput: [batch,3,256,256]')
    X,_ = next(iter(test_loader))
    X = X.to(device)
    print(X.shape)

    print('\nlatent: [batch,16,512]')
    w = E(X)
    print(w.shape)

    print('\nsynth: [batch,3,1024,1024]')
    synth = G.synthesis(E(X))
    print(synth.shape)

    save_image(X, 'tmp/target.png', 1, len(test_set), 256, 256)
    save_image(synth, 'tmp/encoded.png', 1, len(test_set), 256, 256)
