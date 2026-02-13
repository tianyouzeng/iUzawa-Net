import numpy as np
import torch

import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

alpha = 0.01

S = 257

n_test = 2048
batch_size = 1  # keep it 1 for correctly computing sd of loss

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

read_filename = os.path.join(
    root_path, 'data/elliptic/elliptic_f_yd_test_res256_sz2048.npz'
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    f_list, yd_list = data['f'], data['yd']
    data.close()
assert f_list.shape[0] == yd_list.shape[0]

read_filename = os.path.join(
    root_path, 'data/elliptic/elliptic_u_y_test_res256_sz2048.npz'
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    u_list, y_list = data['u'], data['y']
    data.close()
assert u_list.shape[0] == y_list.shape[0]

read_filename = os.path.join(
    root_path, 'data/elliptic/elliptic_ua_ub_test_res256_sz2048.npz'
)
with open(read_filename, 'rb') as file:
    data = np.load(file)
    ua_list, ub_list = data['ua'], data['ub']
    data.close()
assert ua_list.shape[0] == ub_list.shape[0]

delta = 1e-8
is_strictly_between = (u_list >= ua_list + delta) & (u_list <= ub_list - delta)
is_strictly_between_reshaped = is_strictly_between.reshape(u_list.shape[0], -1)
active_samples_mask = ~np.all(is_strictly_between_reshaped, axis=1)
num_active = np.sum(active_samples_mask)

print(f"Number of active samples: {num_active}")
print(f"Proportion of active samples: {num_active / u_list.shape[0]:.4f}")

if num_active > 0:
    # For active samples, calculate proportion of active pixels
    # Active pixel: not strictly between (i.e. in delta neighborhood of boundary)
    prop_active_pixels = np.mean(
        ~is_strictly_between_reshaped[active_samples_mask], axis=1
    )

    print(
        f"Average proportion of active pixels in active samples: {np.mean(prop_active_pixels):.4f}"
    )
