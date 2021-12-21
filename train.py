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
@click.option('--outdir',           help='Where to save the results', metavar='DIR',        required=True)
@click.option('--encoder',          help='Encoder architecture type',                       type=click.Choice(['base','transformer']), required=True)
@click.option('--data',             help='Training data', metavar='[DIR]',                  type=str, required=True)
@click.option('--gpus',             help='Number of GPUs to use', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--batch',            help='Total batch size', metavar='INT',                 type=click.IntRange(min=1), required=True)
@click.option('--generator',        help='Generator pickle to encode',                      required=True) 

# Encoder settings
@click.option('--w_avg',            help='Train delta w from w_avg',                        is_flag=True)
@click.option('--enc_layers',       help='Transformer encoder layers', metavar='INT',       type=click.IntRange(min=1), default=1)

# Validate
@click.option('--valdata',          help='Validation data', metavar='[DIR]',                type=str)

# Training, logging batch steps
@click.option('--training_steps',   help='Total training steps',                            type=click.IntRange(min=1), default=100001)
@click.option('--val_steps',        help='Validation batch steps',                          type=click.IntRange(min=1), default=10000)
@click.option('--print_steps',      help='How often to print logs',                         type=click.IntRange(min=1), default=50)
@click.option('--tb_steps',         help='How often to log to tensorboard?',                type=click.IntRange(min=1), default=50)
@click.option('--img_snshot_steps', help='How often to save image snapshots?',              type=click.IntRange(min=1), default=100)
@click.option('--net_snshot_steps', help='How often to save network snapshots?',            type=click.IntRange(min=1), default=5000)

# Define Loss
@click.option('--lr',               help='Learning rate', metavar='FLOAT',                  type=click.FloatRange(min=0), default=0.001, show_default=True)
@click.option('--l2_lambda',        help='L2 loss multiplier factor', metavar='FLOAT',      type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--lpips_lambda',     help='LPIPS loss multiplier factor', metavar='FLOAT',   type=click.FloatRange(min=0), default=0.8, show_default=True)
@click.option('--id_lambda',        help='ID loss multiplier factor', metavar='FLOAT',      type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--reg_lambda',       help='e4e reg loss multiplier factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--gan_lambda',       help='e4e gan loss multiplier factor', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--edit_lambda',      help='e4e editability lambda', metavar='FLOAT',         type=click.FloatRange(min=0), default=0.0, show_default=True)

# Reproducibility
@click.option('--seed',             help='Random seed', metavar='INT',                      type=click.IntRange(min=0), default=0, show_default=True)

# Dataloader workers
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',      type=click.IntRange(min=1), default=3, show_default=True)

# Resume
@click.option('--resume_pkl',       help='Network pickle to resume training',               default=None, show_default=True)


def main(**kwargs):
    """Main training script
    """
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.

    c.model_architecture = opts.encoder
    c.dataset_dir = opts.data
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch // opts.gpus
    c.generator_pkl = opts.generator

    c.w_avg = opts.w_avg
    c.num_encoder_layers = opts.enc_layers

    c.val_dataset_dir = opts.valdata

    c.training_steps = opts.training_steps
    c.val_steps = opts.val_steps
    c.print_steps = opts.print_steps
    c.tensorboard_steps = opts.tb_steps
    c.image_snapshot_steps = opts.img_snshot_steps
    c.network_snapshot_steps = opts.net_snshot_steps

    c.learning_rate = opts.lr
    c.l2_lambda = opts.l2_lambda
    c.lpips_lambda = opts.lpips_lambda
    c.id_lambda = opts.id_lambda
    c.reg_lambda = opts.reg_lambda
    c.gan_lambda = opts.gan_lambda
    c.edit_lambda = opts.edit_lambda

    c.random_seed = opts.seed
    c.num_workers = opts.workers
    c.resume_pkl = opts.resume_pkl

    # Description string.
    dataset_name = c.dataset_dir.split('/')[-1]
    desc = f'{c.model_architecture:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    # TODO: add resume related description

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
