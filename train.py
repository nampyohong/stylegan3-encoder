"""Train stylegan3 encoder"""

import json
import os
import random
import re
import tempfile

import click
import numpy as np
import torch

import dnnlib
from training import training_loop_encoder
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',      required=True)
@click.option('--cfg',          help='Base configuration',                            type=click.Choice(['base']), required=True)
@click.option('--data',         help='Training data', metavar='[DIR]',                type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',          type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',               type=click.IntRange(min=1), required=True)

# Optional features.
@click.option('--lr',           help='Learning rate', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option('--l2_lambda',    help='L2 loss multiplier factor', metavar='FLOAT',    type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--lpips_lambda', help='LPIPS loss multiplier factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0.8, show_default=True)
@click.option('--id_lambda',    help='ID loss multiplier factor', metavar='FLOAT',    type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                    type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',    type=click.IntRange(min=1), default=3, show_default=True)


def main(**kwargs):
    """Main training script
    """
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.num_gpus = opts.gpus
    c.learning_rate = opts.lr
    c.lambda1 = opts.l2_lambda
    c.lambda2 = opts.lpips_lambda
    c.lambda3 = opts.id_lambda
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus
    c.random_seed = opts.seed
    c.num_workers = opts.workers

    # Description string.
    dataset_name = opts.data.split('/')[-1]
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(opts.outdir):
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    os.makedirs(f'{c.run_dir}/image_snapshots/')
    os.makedirs(f'{c.run_dir}/network_snapshots/')
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
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
    training_loop_encoder.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
