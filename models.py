import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import itertools
from typing import Sequence, Literal

from utils.types import UzawaPtwiseModuleParams, SATypes


einsum_symbols = "bioacdeghjklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LinearFourier(nn.Module):
    """
    Fourier layer, which can be specified as self-adjoint positive (SAP) or not.
    Requires the input and output lying on the same Hilbert space.
    Specifically, they have the same domain and codomain dimensions.
    """

    def __init__(
        self,
        domain_dim: int,
        input_channels: int,
        output_channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        is_sap: bool=False,
        init_param: Literal["default", "identity"]="default",
    ) -> None:
        # TODO: implement different in and out channels
        super().__init__()

        if is_sap:
            assert input_channels == output_channels, "Input and output channels must be equal for self-adjoint positive layers."
        self.input_domain_dim = domain_dim
        self.input_channels = input_channels
        self.output_domain_dim = domain_dim
        self.output_channels = output_channels
        self.modes = modes
        assert len(self.modes) == self.input_domain_dim
        self.is_self_adjoint = is_sap
        self.train_res = train_res
        assert len(self.train_res) == self.input_domain_dim, "Fourier resolution must match the input domain dimension."

        phi_size = (
            [self.input_channels, self.output_channels]
            + [2 * m + 1 for m in self.modes[:-1]]
            + [self.modes[-1] + 1]
        )
        if init_param == "default":
            init_normal_std = (1.0 / (self.input_channels + self.output_channels))
            self.phi = nn.Parameter(torch.normal(0.0, init_normal_std, size=phi_size, dtype=torch.complex64))
            self.local_mat = nn.Parameter(torch.normal(0.0, init_normal_std, size=(self.input_channels, self.output_channels)))
        elif init_param == "identity":
            self.phi = nn.Parameter(torch.ones(phi_size, dtype=torch.complex64))
            self.local_mat = nn.Parameter(torch.eye(self.input_channels, self.output_channels, dtype=torch.float32))

        domain_str = einsum_symbols[-self.input_domain_dim :]
        self.einsum_str_fourier = "bi" + domain_str + ",io" + domain_str + "->bo" + domain_str
        self.einsum_str_local = "bi" + domain_str + ",io->bo" + domain_str

    def _complex_mul_fourier(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str_fourier, x, weights)
    
    def _complex_mul_local(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str_local, x, weights)

    def forward(self, x: torch.Tensor, output_shape: Sequence[int]|None=None) -> torch.Tensor:
        assert x.ndim == len(self.train_res) + 2, "Input tensor must have shape (batch_size, input_channels, d1, d2, ..., dn)."
        assert all(size % res == 0 for res, size in zip(self.train_res, x.shape[2:])), "Invalid input shape or fourier_res."
        
        input_resample_factor = [size // res for res, size in zip(self.train_res, x.shape[2:])]
        input_resample_index = [slice(None, None, rf) for rf in input_resample_factor]
        if output_shape is not None:
            assert output_shape == len(self.train_res)
            assert all(size % res == 0 for res, size in zip(self.train_res, output_shape)), "Invalid output shape or fourier_res."
            output_resample_factor = [size // res for res, size in zip(self.train_res, output_shape)]   # relative to train_res
            output_resample_index = [slice(None, None, rf) for rf in output_resample_factor]    # relative to input shape rather than train_res
        else:
            output_resample_factor = [1] * len(self.train_res)
            output_resample_index = [slice(None)] * len(self.train_res)
        
        x_ft = x[..., *input_resample_index]
        x_ft = torch.fft.rfftn(
            x_ft,
            dim=tuple(range(-self.input_domain_dim, 0)),
            norm='ortho',
        ) # / torch.sqrt(torch.prod(torch.tensor(input_resample_factor, dtype=torch.float32, device=x.device)))
        if output_shape is not None:
            output_ft_shape = x.size()[0:1] + (self.output_channels,) + tuple(output_shape)[:-1] + (output_shape[-1] // 2 + 1,)
        else:
            output_ft_shape = x.size()[0:1] + (self.output_channels,) + x.size()[2:-1] + (x.size(-1) // 2 + 1,)     # (b, o, d1, d2, ..., dn, m)
        output_ft = torch.zeros(output_ft_shape, dtype=torch.cfloat, device=x_ft.device)

        mode_indices = [((None, m + 1), (-m, None)) for m in self.modes[:-1]] + [
            ((None, self.modes[-1] + 1),)
        ]
        pre_index = [slice(None, None, None), slice(None, None, None)]

        if self.is_self_adjoint:
            kernel = self.phi * self.phi.conj().transpose(0, 1)
            mat = torch.matmul(self.local_mat, self.local_mat.transpose(-2, -1))
        else:
            kernel = self.phi
            mat = self.local_mat

        for boundaries in itertools.product(*mode_indices):
            mode_index = [slice(*b) for b in boundaries]
            full_index = pre_index + mode_index
            output_ft[full_index] = self._complex_mul_fourier(
                x_ft[full_index], kernel[full_index]
            )

        if output_shape is not None:
            output_ft = torch.fft.irfftn(
                output_ft,
                dim=tuple(range(-self.input_domain_dim, 0)),
                s=output_shape,
                norm='ortho',
            )
        else:
            output_ft = torch.fft.irfftn(
                output_ft,
                dim=tuple(range(-self.input_domain_dim, 0)),
                norm='ortho',
            )
        output_ft = torch.sqrt(torch.prod(torch.tensor(input_resample_factor, dtype=torch.float32, device=output_ft.device))) * output_ft

        output_local = self._complex_mul_local(x, mat)
        output_local = output_local[..., *output_resample_index]    # TODO

        output = output_ft + output_local
        return output
    

class SAPFourier(nn.Module):

    def __init__(self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        width: int,
        width_pre_lifting: int,
        train_res: Sequence[int],
        padding: Sequence[int]|None=None,
    ):
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.width = width
        self.width_pre_lifting = width_pre_lifting
        self.train_res = train_res
        self.padding = padding

        fourier_layers_train_res = [tr + pd for tr, pd in zip(self.train_res, self.padding)] if self.padding is not None else self.train_res
        self.pre_lifting_mat = nn.Parameter(torch.empty(self.channels, self.width_pre_lifting, dtype=torch.float32))
        self.lifting = nn.Parameter(torch.empty(self.width_pre_lifting, self.width, dtype=torch.float32))
        self.fourier_layer_sap = LinearFourier(
            domain_dim=self.domain_dim,
            input_channels=self.width,
            output_channels=self.width,
            modes=self.modes,
            train_res=fourier_layers_train_res,
            is_sap=True
        )
        nn.init.kaiming_uniform_(self.pre_lifting_mat, a=math.sqrt(5))      # default init as in linear layers in PyTorch 2.7
        nn.init.kaiming_uniform_(self.lifting, a=math.sqrt(5))

        domain_str = einsum_symbols[-self.domain_dim:]
        self.einsum_str = f"b{domain_str}i,io->b{domain_str}o"

    def _mat_mul(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.einsum_str, x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mat_mul(x, self.pre_lifting_mat)
        x = self._mat_mul(x, self.lifting)
        x = x.permute(0, -1, *list(range(1, x.ndim - 1)))

        if self.padding is not None:
            resample_factor = tuple((size - 1) // (res - 1) for res, size in zip(self.train_res, x.shape[2:]))
            padding_size = tuple(rf * (pd + 1) - 1 for rf, pd in zip(resample_factor, self.padding))
            self.pad_info = tuple(dim_info for ps in reversed(padding_size) for dim_info in (0, ps))
            self.unpad_info = tuple(slice(None, -padding) if padding > 0 else slice(None) for padding in padding_size)
            x = F.pad(x, pad=self.pad_info, value=0.0)
        
        x = self.fourier_layer_sap(x)

        if self.padding is not None:
            x = x[..., *self.unpad_info]
        
        x = x.permute(0, *list(range(2, x.ndim)), 1)
        x = self._mat_mul(x, self.lifting.transpose(-2, -1))
        x = self._mat_mul(x, self.pre_lifting_mat.transpose(-2, -1))
        return x
    


class FNO(nn.Module):
    def __init__(
        self,
        domain_dim: int,
        input_channels: int,
        output_channels: int,
        layers: int,
        modes: Sequence[int],
        width: int,
        width_pre_proj: int,
        train_res: Sequence[int],
        padding: Sequence[int]|None=None,
    ) -> None:
        super().__init__()

        assert len(modes) == domain_dim, "Number of modes must match the domain dimension."
        assert padding is None or len(padding) == domain_dim, "Padding must be None or match the domain dimension."
        if padding is not None:
            assert all(p >= 0 for p in padding), "Padding values must be non-negative."

        self.domain_dim = domain_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        self.width = width
        self.width_pre_proj = width_pre_proj
        self.layers = layers
        self.padding = padding
        self.train_res = train_res

        fourier_layers_train_res = [tr + pd for tr, pd in zip(self.train_res, self.padding)] if self.padding is not None else self.train_res
        self.lifting = nn.Linear(self.domain_dim + self.input_channels, self.width)
        self.fourier_layers = nn.ModuleList([LinearFourier(
            domain_dim=domain_dim,
            input_channels=self.width,
            output_channels=self.width,
            modes=modes,
            train_res=fourier_layers_train_res,
            is_sap=False,
        ) for _ in range(layers)])
        self.pre_projection = nn.Linear(self.width, self.width_pre_proj)
        self.projection = nn.Linear(self.width_pre_proj, self.output_channels)
        self.activation = nn.GELU()

    def _get_grid(self, shape, device):
        batch_size, sizes = shape[0], shape[1:-1]
        grids = torch.meshgrid(
            *[torch.linspace(0, 1, size, dtype=torch.float, device=device) for size in sizes],
            indexing='xy'
        )
        grids = [g.reshape(1, *sizes, 1).repeat(batch_size, *(1,) * self.domain_dim, 1) for g in grids]
        return torch.cat(grids, dim=-1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        grid = self._get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.lifting(x)
        x = x.permute(0, -1, *list(range(1, x.ndim - 1)))

        if self.padding is not None:
            resample_factor = tuple((size - 1) // (res - 1) for res, size in zip(self.train_res, x.shape[2:]))
            padding_size = tuple(rf * (pd + 1) - 1 for rf, pd in zip(resample_factor, self.padding))
            self.pad_info = tuple(dim_info for ps in reversed(padding_size) for dim_info in (0, ps))
            self.unpad_info = tuple(slice(None, -padding) if padding > 0 else slice(None) for padding in padding_size)
            x = F.pad(x, pad=self.pad_info, value=0.0)

        for layer in self.fourier_layers:
            x = layer(x)
            x = self.activation(x)

        if self.padding is not None:
            x = x[..., *self.unpad_info]
        x = x.permute(0, *list(range(2, x.ndim)), 1)
        x = self.pre_projection(x)
        x = self.activation(x)
        x = self.projection(x)
        return x
    

class UzawaLayerPtwise(nn.Module):
    """
    Uzawa layer for solving elliptic optimal control problems.
    This layer implements the Uzawa algorithm with self-adjoint positive layers.
    """

    def __init__(self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        alpha: float,
        module_params: UzawaPtwiseModuleParams,
        padding: Sequence[int]|None=None,
        sa_type: SATypes|None=None,
        sol_oper: FNO|None=None,
        apply_prox: bool = True,
    ):
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.alpha = alpha
        self.apply_prox = apply_prox

        self.qb_sap_fourier_layers = SAPFourier(
            domain_dim=domain_dim,
            channels=channels,
            modes=modes,
            width=module_params['qb_width'],
            width_pre_lifting=module_params['qb_width_pre_lift'],
            train_res=train_res,
            padding=padding,
        )
        self.qa_delta = 1.0    # Q_A = (1 + delta) * I     # TODO: make it a parameter

        if sol_oper is None:
            self.S_oper = FNO(
                domain_dim=domain_dim,
                input_channels=channels,
                output_channels=channels,
                layers=module_params['fno_layers'],
                modes=modes,
                width=module_params['fno_width'],
                width_pre_proj=module_params['fno_width_pre_proj'],
                train_res=train_res,
                padding=padding,
            )
            if sa_type == 'sa':
                # TODO: the type of self.S_adj_oper here is inconsistent with the case when sa_type=='sa_time'
                # refactor here later
                self.S_adj_oper = self.S_oper
            elif sa_type == 'sa_time':
                self.S_adj_oper = self._S_adj_oper_sat
            elif sa_type is None or sa_type == 'none':
                self.S_adj_oper = FNO(
                domain_dim=domain_dim,
                input_channels=channels,
                output_channels=channels,
                layers=module_params['fno_layers'],
                modes=modes,
                width=module_params['fno_width'],
                width_pre_proj=module_params['fno_width_pre_proj'],
                train_res=train_res,
                padding=padding,
            )
            else:
                raise ValueError(f"Invalid self-adjoint type: {sa_type}")
        else:
            for param in sol_oper.parameters():
                param.requires_grad = False
            for param in sol_oper.parameters():
                param.requires_grad = False
            sol_oper.eval()
            self.S_oper = sol_oper
            self.S_adj_oper = sol_oper

        if self.apply_prox:
            self.prox_oper = nn.ModuleList([
                nn.Linear(channels + 2, module_params['qa_width']),
                nn.ReLU(),
                nn.Linear(module_params['qa_width'] + 3, module_params['qa_width']),
                nn.ReLU(),
                nn.Linear(module_params['qa_width'] + 3, channels),
            ])

    def _S_adj_oper_sa(self, x: torch.Tensor) -> torch.Tensor:
        return self.S_oper(x)

    def _S_adj_oper_sat(self, x: torch.Tensor) -> torch.Tensor:
        return self.S_oper(x.flip([-2])).flip([-2])

    def _prox_oper(
        self,
        input: torch.Tensor,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
    ) -> torch.Tensor:
        u = torch.cat((input, u_a, u_b), dim=-1)
        u = self.prox_oper[0](u)
        u = self.prox_oper[1](u)
        u = torch.cat((u, u_a, u_b, input), dim=-1)
        u = self.prox_oper[2](u)
        u = self.prox_oper[3](u)
        u = torch.cat((u, u_a, u_b, input), dim=-1)
        u = self.prox_oper[4](u)
        return u

    def _u_step(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
    ) -> torch.Tensor:
        u_new = self.qa_delta / (1.0 + self.qa_delta) * u - (1.0 / ((1.0 + self.qa_delta) * self.alpha)) * self.S_adj_oper(p)
        if self.apply_prox:
            u_new = self._prox_oper(u_new, u_a, u_b)
        return u_new
    
    def _p_step(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        yd: torch.Tensor,
    ) -> torch.Tensor:
        p_step = self.qb_sap_fourier_layers(self.S_oper(u + f) - p - yd)
        p_new = p + p_step
        return p_new

    def forward(
        self,
        u: torch.Tensor,
        p: torch.Tensor,
        yd: torch.Tensor,
        f: torch.Tensor,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        u = self._u_step(u, p, u_a, u_b)
        p = self._p_step(u, p, f, yd)

        return u, p
    

class DUUzawaShared(nn.Module):
    def __init__(
        self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        layers: int,
        alpha: float,
        layer_module_params: UzawaPtwiseModuleParams,
        padding: Sequence[int]|None=None,
        sa_type: SATypes|None=None,
        sol_oper: FNO|None=None,
    ) -> None:
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.layers = layers
        self.alpha = alpha

        self.uzawa_layer = UzawaLayerPtwise(
            domain_dim=domain_dim,
            channels=channels,
            modes=modes,
            alpha=alpha,
            sol_oper=sol_oper,
            train_res=train_res,
            module_params=layer_module_params,
            padding=padding,
            sa_type=sa_type,
        )

    def forward(
        self,
        yd: torch.Tensor,
        f: torch.Tensor,
        u_a: float,
        u_b: float,
    ) -> torch.Tensor:    # type: ignore
        u = torch.zeros_like(yd)
        p = torch.zeros_like(yd)

        for _ in range(self.layers):
            u, p = self.uzawa_layer(u, p, yd, f, u_a, u_b)

        return u


class DUUzawaFull(nn.Module):
    def __init__(
        self,
        domain_dim: int,
        channels: int,
        modes: Sequence[int],
        train_res: Sequence[int],
        layers: int,
        alpha: float,
        layer_module_params: UzawaPtwiseModuleParams,
        padding: Sequence[int]|None=None,
        sa_type: SATypes|None=None,
        sol_oper: FNO|None=None,
    ) -> None:
        super().__init__()

        self.domain_dim = domain_dim
        self.channels = channels
        self.modes = modes
        self.layers = layers
        self.alpha = alpha

        self.uzawa_layers = nn.ModuleList([UzawaLayerPtwise(
            domain_dim=domain_dim,
            channels=channels,
            modes=modes,
            alpha=alpha,
            module_params=layer_module_params,
            sa_type=sa_type,
            sol_oper=sol_oper,
            train_res=train_res,
            padding=padding,
        ) for _ in range(layers)])

    def forward(
        self,
        yd: torch.Tensor,
        f: torch.Tensor,
        u_a: float,
        u_b: float,
    ) -> torch.Tensor:    # type: ignore
        u = torch.zeros_like(yd)
        p = torch.zeros_like(yd)

        for layer in self.uzawa_layers:
            u, p = layer(u, p, yd, f, u_a, u_b)

        return u
    