import numpy as np
import torch
import math
import random
import warnings

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from . import _base

class Reservoir(torch.nn.Module):

    def __init__(self, input_dim, edge_length, 
            liquid_filter: _base.FilterBase = None,
            voltage_decay = 0.7,
            basic_threshold = 10,
            threshold_gain_decay = 0.995,
            threshold_gain_ratio = 0.07
            ):
        '''
        In kwargs:
        For liquid state
        1. liquid_filter_setup: (filter class, param)
        For neurons:
        1. voltage_decay: default 0.5
        2. basic_threshold: default 10
        3. threshold_gain_decay: default 0.995
        4. threshold_gain_ratio: default 0.07
        '''
        super().__init__()
        self.input_dim = input_dim
        self.edge_length = edge_length

        self.liquid_filter: _base.FilterBase = liquid_filter
        if liquid_filter is None:
            self.liquid_filter = _base.LowPassFilter(0.33)

        ALIF_kwargs = dict()
        ALIF_kwargs["voltage_decay"] = voltage_decay
        ALIF_kwargs["basic_threshold"] = basic_threshold
        ALIF_kwargs["threshold_gain_decay"] = threshold_gain_decay
        ALIF_kwargs["threshold_gain_ratio"] = threshold_gain_ratio
        self.somas = _base.ALIF(**ALIF_kwargs)

        self.input_weights = torch.nn.Parameter(
                torch.zeros((edge_length**3, input_dim), dtype=torch.float32),
                requires_grad = False
                )
        self.recursive_weights = torch.nn.Parameter(
                torch.zeros((edge_length**3, edge_length**3), dtype=torch.float32),
                requires_grad = False
                )

        self.generate_recursive_weights()
        self.generate_input_weights()

    def clone(self):
        new_reservoir = Reservoir(
                self.input_dim, self.edge_length,
                voltage_decay = self.somas.voltage_decay,
                basic_threshold = self.somas.basic_threshold,
                threshold_gain_decay = self.somas.threshold_gain_decay,
                threshold_gain_ratio = self.somas.threshold_gain_ratio
                )
        new_reservoir.input_weights.data = self.input_weights.data.clone()
        new_reservoir.recursive_weights.data = self.recursive_weights.clone()
        return new_reservoir

    def idx2coordinate(self, idx):
        x = idx // (self.edge_length**2)
        idx -= x * self.edge_length**2
        y = idx // self.edge_length
        z = idx - y * self.edge_length
        return (x,y,z)

    def indices2distance(self, idx_1, idx_2):
        x_1, y_1, z_1 = self.idx2coordinate(idx_1)
        x_2, y_2, z_2 = self.idx2coordinate(idx_2)
        distance = math.sqrt((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)
        return distance

    @torch.jit.ignore
    def generate_recursive_weights(self,
            C = 0.2,
            sigma = 2.6,
            mean = 2.0,
            std = 0.2,
            inhibitory_ratio = 0.4
            ):
        r'''
        $$
        P(i, j) = C \cdot \exp(-\frac{D(i,j)^2}{\sigma^2})
        $$
        where $std=\sigma$
        kwargs:
        1. C: default 0.2
        2. sigma: default 3
        3. mean: default 0.1
        4. std: default 0.2
        5. inhibitory_ratio: 0.2
        '''
        gaussian_weights = torch.randn(*self.recursive_weights.shape)*std + mean
        gaussian_weights[gaussian_weights<0] = 0
        
        distance_mat = torch.zeros_like(self.recursive_weights)
        m = 0
        while m < self.edge_length:
            n = m + 1
            while n < self.edge_length:
                distance = self.indices2distance(m, n)
                distance_mat[m, n] = distance
                distance_mat[n, m] = distance
                n += 1
            m += 1

        probability_mat = C * torch.exp(-distance_mat**2/sigma**2)
        connect_standard = torch.rand(*self.recursive_weights.shape)
        disconnect_mask = (connect_standard > probability_mat) # Is 1 if no connection
        for i in range(self.edge_length):
            disconnect_mask[i, i] = 1 # Concel the self-loop
        gaussian_weights[disconnect_mask] = 0
        recursive_weights = gaussian_weights

        inhibitory_num = int(self.edge_length**3 * inhibitory_ratio)
        inhibitory_indices = random.sample(range(self.edge_length**3), k=inhibitory_num)
        recursive_weights[:,inhibitory_indices] *= -1 # Inhibitory Neuron
        self.recursive_weights.data = recursive_weights

    @torch.jit.ignore
    def generate_input_weights(self, **kwargs):
        '''
        kwargs:
        1. density: default 0.3(Ivanov 2021)
        2. mean: default somas.basic_threshold/(input_dim*density)
        3. std: default 0.6
        '''
        density = kwargs.get("density", 0.3)
        mean = kwargs.get("mean", self.somas.basic_threshold/(self.input_dim*density))
        std = kwargs.get("std", 0.6)

        input_gaussian_weights = torch.randn(self.edge_length**3, self.input_dim)*std + mean
        if torch.cuda.is_available():
            input_gaussian_weights = input_gaussian_weights.cuda()
        input_gaussian_weights[input_gaussian_weights<0] = 0
        input_gaussian_weights = input_gaussian_weights.view(-1)
        drop_indices = random.sample(range(len(input_gaussian_weights)), k=int((1-density)*len(input_gaussian_weights)))
        input_gaussian_weights[drop_indices] = 0
        input_weights = input_gaussian_weights.view(self.edge_length**3, self.input_dim)
        self.input_weights.data = input_weights

    @torch.jit.export
    def forward(self, input_spikes):
        '''
        input_spike: [n, batch_size], i.e. using column vector convention
        '''
        input_voltage = torch.matmul(self.input_weights, input_spikes)
        recursive_voltage = torch.matmul(self.recursive_weights, self.somas.get_previous_spike(input_voltage))
        voltage_gain = recursive_voltage + input_voltage
        spike = self.somas(voltage_gain)
        liquid_state = self.liquid_filter(spike)
        return liquid_state

    @torch.jit.export
    def reset(self):
        self.somas.reset()
        self.liquid_filter.reset()

    @torch.jit.export
    def get_output_dim(self):
        return int(self.edge_length ** 3)

class FloatInputReservoir(Reservoir):

    def __init__(self, input_dim, edge_length, 
            liquid_filter_setup = (_base.LowPassFilter, 0.33),
            voltage_decay = 0.7,
            basic_threshold = 10,
            threshold_gain_decay = 0.995,
            threshold_gain_ratio = 0.07,
            spike_encoder: _base.ALIF = None
            ):
        super().__init__(
                input_dim, edge_length, 
                liquid_filter_setup,
                voltage_decay,
                basic_threshold,
                threshold_gain_decay,
                threshold_gain_ratio
                )
        if spike_encoder is None:
            self.spike_encoder = _base.ALIF(voltage_decay=0.9, basic_threshold=10)
        else:
            self.spike_encoder = spike_encoder

    def forward(self, input_floats):
        input_spikes = self.spike_encoder(input_floats)
        input_voltage = torch.matmul(self.input_weights, input_spikes)
        recursive_voltage = torch.matmul(self.recursive_weights, self.somas.get_previous_spike(input_voltage))
        voltage_gain = recursive_voltage + input_voltage
        spike = self.somas(voltage_gain)
        liquid_state = self.liquid_filter(spike)
        return liquid_state

    @torch.jit.export
    def reset(self):
        self.somas.reset()
        self.liquid_filter.reset()
        self.spike_encoder.reset()

class ReservoirOfflineWrapper(_base.OfflineWrapperBase):

    def __init__(self, reservoir):
        super().__init__(reservoir)
    
    def get_output_shape(self, input_channel: int):
        dim = self.system.get_output_dim() # For LSM, the output shape is fixed after initialization
        return (dim,)

def compute_lyapunov_exponent(reservoir: Reservoir, test_num=10):
    if torch.cuda.is_available():
        reservoir = reservoir.cuda()

    lyapunovs = []

    for i in range(test_num):
        spike_train = _base.generate_spike_train(reservoir.input_dim, 20, 0.4)
        spike_indices = [] # (timestep, channel)
        for timestep in range(spike_train.shape[-1]):
            for channel in range(spike_train.shape[0]):
                value = spike_train[channel, timestep]
                if value == 1:
                    spike_indices.append((timestep, channel))
                    if len(spike_indices) == 2:
                        break
            if len(spike_indices) == 2:
                break

        spike_train_var = spike_train.clone()
        var_timestep, var_channel = spike_indices[0]
        if var_timestep != 0:
            spike_train_var[var_channel, 0] = 1
        spike_train_var[var_channel, var_timestep] = 0
        batch = torch.cat([spike_train.unsqueeze(0), spike_train_var.unsqueeze(0)], dim=0)

        wrapper = ReservoirOfflineWrapper(reservoir)
        liquid_states = wrapper(batch)
        truncate_time, _ = spike_indices[1]
        truncated_liquid_states = liquid_states[..., truncate_time+1:]

        variation = (truncated_liquid_states[0] - truncated_liquid_states[1])**2
        variation = torch.sqrt(variation.sum(dim=0)) # [timestep,]
        if variation[0].item() < 1e-9:
            warnings.warn("(For Lyapunov exponent) The reservoir is not sensitive enough to the single-spike variation.")
        var_ratio = variation / variation[0]
        epsilon = 1e-10
        log_var = torch.log(var_ratio+epsilon).cpu().numpy()

        timesteps = np.arange(len(log_var))
        exponent, _ = np.polyfit(timesteps, log_var, 1)
        lyapunovs.append(exponent)

    lyapunovs_arr = np.array(lyapunovs)
    lyapunov_exponent = lyapunovs_arr.mean()
    return lyapunov_exponent

class StdLSM(torch.nn.Module):

    def __init__(self, reservoir: Reservoir, output_dim = 2, random_ratio = 0.1):
        super().__init__()
        self.random_ratio = random_ratio
        self.wrapped_reservoir = ReservoirOfflineWrapper(reservoir)
        self.wrapped_reservoir = torch.jit.script(self.wrapped_reservoir)
        dim_mid = reservoir.get_output_dim()
        self.regression = torch.nn.Linear(dim_mid, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.wrapped_reservoir(x)
            if self.training:
                end = x.shape[-1]
                start = int(self.random_ratio * end)
                idx = np.random.randint(start, end)
                x = x[..., idx]
            else:
                x = x[..., -1]
        x = self.regression(x)
        return x

# Following: Some validated realization of reservoirs

def get_in14_edge7_avgfilter_reservoir():
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.00064 (300 times sampling).
    '''
    channel = 14
    edge_length = 7
    avg_filter = _base.IterativeMovingAverage(0.21)
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    reservoir.generate_recursive_weights(
            C = 0.12,
            sigma = 4.0,
            mean = 0.24,
            std = 0.15,
            inhibitory_ratio = 0.3)
    reservoir.generate_input_weights(
            density = 0.15)
    lyap = compute_lyapunov_exponent(reservoir)
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
    
def get_in3_edge7_avgfilter_reservoir():
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0037.
    '''
    channel = 3
    edge_length = 7
    avg_filter = _base.IterativeMovingAverage(0.1)
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    reservoir.generate_recursive_weights(
            C = 0.35,
            sigma = 1,
            mean = 0.25,
            std = 0.05,
            inhibitory_ratio = 0.33)
    reservoir.generate_input_weights(
            density = 0.17)
    lyap = compute_lyapunov_exponent(reservoir)
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
    
def get_in62_edge9_avgfilter_reservoir():
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0037.
    '''
    channel = 62
    edge_length = 9
    avg_filter = _base.IterativeMovingAverage(0.1)
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    reservoir.generate_recursive_weights(
            C = 0.2,
            sigma = 1,
            mean = 0.22,
            std = 0.05,
            inhibitory_ratio = 0.3)
    reservoir.generate_input_weights(
            density = 0.033)
    lyap = compute_lyapunov_exponent(reservoir)
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
