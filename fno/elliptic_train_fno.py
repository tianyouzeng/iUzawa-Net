import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import os
from timeit import default_timer

from models import FNO
from utils.utils_fno import LpLoss

torch.manual_seed(114514)
np.random.seed(114514)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

S = 65
T = 65

n_train = 16384
batch_size = 64
epochs = 300
learning_rate = 0.001
scheduler_step = 30
scheduler_gamma = 0.6

read_filename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data/elliptic/elliptic_f_yd_train_res64_sz16384.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    f_list, yd_list = data['f'], data['yd']
    data.close()
assert f_list.shape[0] == yd_list.shape[0]

read_filename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data/elliptic/elliptic_u_y_train_res64_sz16384.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    u_list, y_list = data['u'], data['y']
    data.close()
assert u_list.shape[0] == y_list.shape[0]

read_filename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data/elliptic/elliptic_ua_ub_train_res64_sz16384.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    ua_list, ub_list = data['ua'], data['ub']
    data.close()
assert ua_list.shape[0] == ub_list.shape[0]

u_train = torch.tensor(u_list, dtype=torch.float32)[:n_train].to(device)
f_train = torch.tensor(f_list, dtype=torch.float32)[:n_train].to(device)
yd_train = torch.tensor(yd_list, dtype=torch.float32)[:n_train].to(device)
y_train = torch.tensor(y_list, dtype=torch.float32)[:n_train].to(device)
ua_train = torch.tensor(ua_list, dtype=torch.float32)[:n_train].to(device)
ub_train = torch.tensor(ub_list, dtype=torch.float32)[:n_train].to(device)

u_train = u_train.reshape(n_train, S, S, 1)
f_train = f_train.reshape(n_train, S, S, 1)
yd_train = yd_train.reshape(n_train, S, S, 1)
y_train = y_train.reshape(n_train, S, S, 1)
ua_train = ua_train.reshape(n_train, S, S, 1)
ub_train = ub_train.reshape(n_train, S, S, 1)

train_loader = DataLoader(
    TensorDataset(u_train, f_train, yd_train, y_train, ua_train, ub_train),
    batch_size=batch_size,
    shuffle=True,
)

# Parameters number similar to DUUzawaShared
model = FNO(
    domain_dim=2,
    input_channels=4,
    output_channels=1,
    layers=3,
    modes=(10, 10),
    width=8,
    width_pre_proj=128,
    train_res=(65, 65),
    padding=(7, 7),
).to(device)
model.train()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=scheduler_step, gamma=scheduler_gamma
)
loss_func = LpLoss(size_average=False)

for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0

    for u, f, yd, y, ua, ub in train_loader:
        optimizer.zero_grad()
        out = model(torch.cat([yd, f, ua, ub], dim=-1))
        mse = F.mse_loss(out, u, reduction='mean')
        l2 = loss_func(out.view(batch_size, -1), u.view(batch_size, -1))
        l2.backward()
        optimizer.step()

        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    train_mse /= len(train_loader)
    train_l2 /= n_train

    t2 = default_timer()
    print(
        f'epoch: {ep}, time: {t2-t1}, training mse: {train_mse}, training l2: {train_l2}'
    )

torch.save(model.state_dict(), 'elliptic_fno_layer_4.pt')
