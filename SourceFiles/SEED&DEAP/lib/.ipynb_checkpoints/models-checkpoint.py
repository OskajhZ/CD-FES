from . import _base
from . import reservoir
import torch
import numpy as np

class ThreeBandLsmWithConvReadout(torch.nn.Module):

    def __init__(self,
            configured_reservoir: reservoir.Reservoir,
            category_num = 3,
            random_ratio = 0.1
            ):
        '''
        category_num <= 3
        '''
        super().__init__()
        self.random_ratio = random_ratio
        band_amount = 3 # For Theta, Alpha and Beta
        self.band_amount = band_amount

        self.reservoir_edge = configured_reservoir.edge_length
        self.wrapped_reservoirs = torch.nn.ModuleList()
        for _ in range(band_amount):
            sub_reservoir = configured_reservoir.clone()
            wrapped = reservoir.ReservoirOfflineWrapper(sub_reservoir)
            wrapped = torch.jit.script(wrapped)
            self.wrapped_reservoirs.append(wrapped)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parallel_streams = [torch.cuda.Stream(device=device) for _ in range(band_amount)]

        conv_list = []
        residual_edge = self.reservoir_edge
        while residual_edge > 1:
            conv_block = torch.nn.Sequential(
                    torch.nn.Conv3d(3, 3, 3),
                    torch.nn.BatchNorm3d(3),
                    torch.nn.GELU()
                    )
            conv_list.append(conv_block)
            residual_edge -= 2
        self.conv = torch.nn.Sequential(
                *tuple(conv_list)
                )

        self.classifier = torch.nn.Linear(3, category_num)

    def forward(self, x, message=None):
        liquid_state_idx = -1
        if self.training:
            end = x.shape[-1]
            start = int(self.random_ratio * end)
            liquid_state_idx = np.random.randint(start, end)

        x = x[..., :3, :, :]
        reservoir_out = [None for _ in range(self.band_amount)]
        for idx in range(self.band_amount):
            with torch.cuda.stream(self.parallel_streams[idx]):
                with torch.no_grad():
                    out = self.wrapped_reservoirs[idx](x[:, idx]) # [batchsize, reservoir_dim, timestep]
                out = out[..., -1]
                out = out.view(out.shape[0], 1,
                        self.reservoir_edge, self.reservoir_edge, self.reservoir_edge)
                reservoir_out[idx] = out
        torch.cuda.synchronize()
        x = torch.cat(reservoir_out, dim = 1) # [batchsize, channel=3, dim, dim, dim]
        x = self.conv(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
