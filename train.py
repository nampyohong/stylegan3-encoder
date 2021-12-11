"""Train stylegan3 encoder"""

import os
import random
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
#@click.option('--local_rank',   help='Local rank', metavar='INT',                     type=click.IntRange(min=0), required=True)

# Optional features.
@click.option('--lr',           help='Learning rate', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option('--l2_lambda',    help='L2 loss multiplier factor', metavar='FLOAT',    type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--lpips_lambda', help='LPIPS loss multiplier factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0.8, show_default=True)
@click.option('--id_lambda',    help='ID loss multiplier factor', metavar='FLOAT',    type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                    type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',    type=click.IntRange(min=1), default=3, show_default=True)


def main(**kwargs):
    """
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.num_gpus = opts.gpus
    c.learning_rate = opts.lr
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus
    c.random_seed = opts.seed
    c.num_workers = opts.workers

    # Set random seed for reproducability
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
