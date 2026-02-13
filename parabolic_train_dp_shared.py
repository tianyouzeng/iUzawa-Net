import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from timeit import default_timer

from models import DUUzawaFull, DUUzawaShared
from utils.utils_fno import LpLoss
from utils.types import UzawaPtwiseModuleParams

torch.manual_seed(114514)
torch.cuda.manual_seed_all(114514)
np.random.seed(114514)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    alpha = 0.01

    S = 33
    T = 33

    n_train = 16384
    batch_size = 16
    epochs = 300
    learning_rate = 0.002
    scheduler_step = 30
    scheduler_gamma = 0.6

    read_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data/parabolic/parabolic_f_yd_train_res32_sz16384.npz',
    )
    with open(read_filename, 'rb') as file:
        data = np.load(file)
        f_list, yd_list = data['f'], data['yd']
        data.close()

    read_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data/parabolic/parabolic_u_y_train_res32_sz16384.npz',
    )
    with open(read_filename, 'rb') as file:
        data = np.load(file)
        u_list, y_list = data['u'], data['y']
        data.close()

    u_train = torch.tensor(u_list, dtype=torch.float32)[:n_train]
    f_train = torch.tensor(f_list, dtype=torch.float32)[:n_train]
    yd_train = torch.tensor(yd_list, dtype=torch.float32)[:n_train]
    y_train = torch.tensor(y_list, dtype=torch.float32)[:n_train]

    u_train = u_train.reshape(n_train, S, S, T, 1)
    f_train = f_train.reshape(n_train, S, S, T, 1)
    yd_train = yd_train.reshape(n_train, S, S, T, 1)
    y_train = y_train.reshape(n_train, S, S, T, 1)

    ua = -6.0 * torch.ones((batch_size, S, S, T, 1))
    ub = 6.0 * torch.ones((batch_size, S, S, T, 1))

    train_data = TensorDataset(u_train, f_train, yd_train, y_train)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
    )

    layer_module_params: UzawaPtwiseModuleParams = {
        'qa_width': 64,
        'qb_width': 8,
        'qb_width_pre_lift': 64,
        'fno_layers': 3,
        'fno_width': 8,
        'fno_width_pre_proj': 64,
    }
    model = DUUzawaShared(
        domain_dim=3,
        channels=1,
        modes=(8, 8, 8),
        train_res=(33, 33, 33),
        padding=(3, 3, 3),
        layers=5,
        alpha=alpha,
        layer_module_params=layer_module_params,
        sa_type='sa_time',
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    if dist.get_rank() == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    loss_func = LpLoss(size_average=False)

    for ep in range(epochs):
        train_sampler.set_epoch(ep)
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0

        for i, (u, f, yd, y) in enumerate(train_loader):
            optimizer.zero_grad()
            u, f, yd, y, ua, ub = (
                u.to(rank),
                f.to(rank),
                yd.to(rank),
                y.to(rank),
                ua.to(rank),
                ub.to(rank),
            )
            out = model(yd, f, ua, ub)
            mse = F.mse_loss(out, u, reduction='mean')
            l2 = loss_func(out.view(batch_size, -1), u.view(batch_size, -1))
            l2.backward()
            optimizer.step()

            train_mse += mse.detach()
            train_l2 += l2.detach()

        scheduler.step()

        dist.all_reduce(train_l2, op=dist.ReduceOp.SUM)
        train_mse /= len(train_loader)
        train_l2 /= n_train

        t2 = default_timer()
        if dist.get_rank() == 0:
            print(
                f'epoch: {ep}, time: {t2-t1:.4f}, training mse: {train_mse:.10f}, training l2: {train_l2:.10f}'
            )

    dist.barrier()
    if dist.get_rank() == 0:
        torch.save(
            ddp_model.module.state_dict(), 'parabolic_du_uzawa_shared_layer_5_dp.pt'
        )

    cleanup()


if __name__ == '__main__':
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)  # type: ignore
