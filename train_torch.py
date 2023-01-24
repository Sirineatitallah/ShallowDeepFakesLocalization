import os

import torch
import torch.nn as nn

# multiprocessing
from utils.dist import *

from train_base import *

# constants
SYNC = False
GET_MODULE = True

def main():
    args = parse_args()

    # Init dist
    init_dist('slurm', 13721)

    global_rank, world_size = get_dist_info()

    args, checkpoint_dir = init_env_multi(args, global_rank)

    # models
    model = init_models(args)
    model = load_dicts(args, model)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    optimizer = init_optims(args, world_size, model)

    lr_scheduler = init_schedulers(args, optimizer)

    if (args.cond_state is not None):
        saved_state = torch.load(args.cond_state)

        optimizer.load_state_dict(saved_state['optimizer'])
        lr_scheduler.load_state_dict(saved_state['lr_scheduler'])

        prev_best_val_loss = saved_state['best_val_loss']

        # move to device
        device = torch.device('cuda:' + str(torch.cuda.current_device()))

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        prev_best_val_loss = None

    # dataset
    train_sampler, dataloader = init_dataset(args, global_rank, world_size, False)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True)

    train(args, global_rank, world_size, SYNC, GET_MODULE,
            checkpoint_dir,
            model,
            train_sampler, dataloader, val_sampler, val_dataloader,
            optimizer,
            lr_scheduler,
            prev_best_val_loss)

if __name__ == '__main__':
    main()