import torch
import matplotlib.pyplot as plt


def plot_result_parab_grid(
    u: torch.Tensor,
    out: torch.Tensor,
    instance_idx: int,
    time_idx: tuple[int, ...],
    cb_range: tuple[float, float],
) -> None:
    '''Plot and save the results for a specific instance and time indices.'''
    for time in time_idx:
        plt.imshow(
            out[0, :, :, time, 0].cpu().numpy(),
            cmap='coolwarm',
            extent=(0, 1, 0, 1),
            vmin=cb_range[0],
            vmax=cb_range[1],
        )
        plt.colorbar()
        plt.savefig(f'parabolic_u_computed_{instance_idx}_{time}.pdf')
        plt.show()
        plt.close()

        plt.imshow(
            u[0, :, :, time, 0].cpu().numpy(),
            cmap='coolwarm',
            extent=(0, 1, 0, 1),
            vmin=cb_range[0],
            vmax=cb_range[1],
        )
        plt.colorbar()
        plt.savefig(f'parabolic_u_exact_{instance_idx}_{time}.pdf')
        plt.show()
        plt.close()

        diff = u[0, :, :, time, 0].cpu().numpy() - out[0, :, :, time, 0].cpu().numpy()
        plt.imshow(diff, cmap='coolwarm', extent=(0, 1, 0, 1))
        plt.colorbar()
        plt.savefig(f'parabolic_u_diff_{instance_idx}_{time}.pdf')
        plt.show()
        plt.close()
