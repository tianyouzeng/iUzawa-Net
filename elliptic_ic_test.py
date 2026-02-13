import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import os
from timeit import default_timer

from models import DUUzawaFull
from utils.utils_fno import LpLoss
from utils.types import UzawaPtwiseModuleParams

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

alpha = 0.01

S = 65

n_test = 2048
batch_size = 1  # keep it 1 for correctly computing sd of loss

read_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data/elliptic_ic/elliptic_ic_f_yd_test_res64_sz2048.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    f_list, yd_list = data['f'], data['yd']
    data.close()
assert f_list.shape[0] == yd_list.shape[0]

read_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data/elliptic_ic/elliptic_ic_u_y_test_res64_sz2048.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    u_list, y_list = data['u'], data['y']
    data.close()
assert u_list.shape[0] == y_list.shape[0]

read_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data/elliptic_ic/elliptic_ic_ua_ub_test_res64_sz2048.npz',
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    ua_list, ub_list = data['ua'], data['ub']
    data.close()
assert ua_list.shape[0] == ub_list.shape[0]

u_test = torch.tensor(u_list, dtype=torch.float32)[:n_test].to(device)
f_test = torch.tensor(f_list, dtype=torch.float32)[:n_test].to(device)
yd_test = torch.tensor(yd_list, dtype=torch.float32)[:n_test].to(device)
y_test = torch.tensor(y_list, dtype=torch.float32)[:n_test].to(device)
ua_test = torch.tensor(ua_list, dtype=torch.float32)[:n_test].to(device)
ub_test = torch.tensor(ub_list, dtype=torch.float32)[:n_test].to(device)

u_test = u_test.reshape(n_test, S, S, 1)
f_test = f_test.reshape(n_test, S, S, 1)
yd_test = yd_test.reshape(n_test, S, S, 1)
y_test = y_test.reshape(n_test, S, S, 1)
ua_test = ua_test.reshape(n_test, S, S, 1)
ub_test = ub_test.reshape(n_test, S, S, 1)

test_loader = DataLoader(
    TensorDataset(u_test, f_test, yd_test, y_test, ua_test, ub_test),
    batch_size=batch_size,
    shuffle=False,
)

layer_module_params: UzawaPtwiseModuleParams = {
    'qa_width': 64,
    'qb_width': 8,
    'qb_width_pre_lift': 64,
    'fno_layers': 3,
    'fno_width': 8,
    'fno_width_pre_proj': 64,
}
model = DUUzawaFull(
    domain_dim=2,
    channels=1,
    modes=(8, 8),
    train_res=(65, 65),
    padding=(7, 7),
    layers=6,
    alpha=alpha,
    layer_module_params=layer_module_params,
    sa_type='sa',
).to(device)
model.load_state_dict(torch.load('trained_models/elliptic_ic_du_uzawa_layer_6.pt'))
model.to(device)
model.eval()

loss_func = LpLoss(size_average=False)

time_start = default_timer()

with torch.no_grad():
    mse_test = torch.zeros(n_test)
    l2_test = torch.zeros(n_test)
    l2_test_abs = torch.zeros(n_test)

    for i, (u, f, yd, y, u_a, u_b) in enumerate(
        tqdm(
            test_loader, total=n_test // batch_size, bar_format="{l_bar}{bar:36}{r_bar}"
        )
    ):
        out = model(yd, f, u_a, u_b)
        mse = F.mse_loss(out, u, reduction='mean')
        l2 = loss_func(out.reshape(batch_size, -1), u.reshape(batch_size, -1))
        l2_abs = torch.norm(out - u) / (S - 1)
        mse_test[i] = mse.item()
        l2_test[i] = l2.item()
        l2_test_abs[i] = l2_abs.item()

    mse_test_mean = torch.mean(mse_test)
    l2_test_mean = torch.mean(l2_test)
    mse_test_sd = torch.std(mse_test)
    l2_test_sd = torch.std(l2_test)
    l2_test_abs_mean = torch.mean(l2_test_abs)
    l2_test_abs_sd = torch.std(l2_test_abs)

time_end = default_timer()

print(f'test mse: {mse_test_mean}, test l2: {l2_test_mean}')
print(f'test mse sd: {mse_test_sd}, test l2 sd: {l2_test_sd}')
print(f'test l2 abs: {l2_test_abs_mean}, test l2 abs sd: {l2_test_abs_sd}')
print(
    f'test time: {time_end - time_start} seconds, mean: {(time_end - time_start) / n_test} seconds per sample'
)
