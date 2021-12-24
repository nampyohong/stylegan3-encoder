"""Test stylegan3 encoder"""

import json
import os
import random
import re
import tempfile

import click
import numpy as np
import torch

import dnnlib
from training import testing_loop_encoder
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--testdir',          help='Training exp directory path', metavar='DIR',      required=True)
@click.option('--data',             help='Testing data', metavar='[DIR]',                   type=str, required=True)
@click.option('--gpus',             help='Number of GPUs to use', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--batch',            help='Total batch size', metavar='INT',                 type=click.IntRange(min=1), required=True)

# Reproducibility
@click.option('--seed',             help='Random seed', metavar='INT',                      type=click.IntRange(min=0), default=0, show_default=True)

# Dataloader workers
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',      type=click.IntRange(min=1), default=3, show_default=True)


def main(**kwargs):
    """Main training script
    """
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.

    c.test_dir = opts.testdir
    c.dataset_dir = opts.data
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus

    c.random_seed = opts.seed
    c.num_workers = opts.workers

    with open(os.path.join(c.test_dir, 'training_options.json'),'r') as f:
        training_options = dnnlib.EasyDict(json.load(f))

    c.model_architecture = training_options.model_architecture
    if 'w_avg' in training_options:
        c.w_avg = training_options.w_avg
    if 'num_encoder_layers' in training_options:
        c.num_encoder_layers = training_options.num_encoder_layers
    c.generator_pkl = training_options.generator_pkl
    c.l2_lambda = training_options.l2_lambda
    c.lpips_lambda = training_options.lpips_lambda
    c.id_lambda = training_options.id_lambda
    c.reg_lambda = training_options.reg_lambda
    c.gan_lambda = training_options.gan_lambda
    c.edit_lambda = training_options.edit_lambda

    # Print options.
    print()
    print('Testing options:')
    print(json.dumps(c, indent=2))
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(f'{c.test_dir}/test', exist_ok=True)
    with open(os.path.join(c.test_dir, 'test', 'testing_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


def subprocess_fn(rank, c, temp_dir):
    # Init torch.distributed.
#    if c.num_gpus > 1:
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    init_method = f'file://{init_file}'
    torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils
    torch.cuda.set_device(rank)
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    testing_loop_encoder.testing_loop(rank=rank, **c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
