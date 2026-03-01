'''
By Xiangnan Zhang, 2025
School of Future Technologies, Beijing Institute of Technology.
Version for the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition
'''

import torch

class LSMBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.reservoir: torch.nn.Module = None
        self.readout: torch.nn.Module = None

    def register(self, reservoir: torch.nn.Module, readout: torch.nn.Module):
        self.reservoir = reservoir
        self.readout = readout

    def get_reservoir(self):
        return self.reservoir

    def get_readout(self):
        return self.readout

# Following: LSTM benchmark

class LSTMStream(torch.nn.Module):

    def __init__(self, original_channels, out_dim):
        super().__init__()

        self.lstm = torch.nn.LSTM(
                input_size = original_channels,
                hidden_size = out_dim,
                num_layers = 1,
                batch_first = True
                )
        self.norm = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        '''
        x: [batch_size, channels, timesteps]
        return: [batch_size, output_dim]
        '''
        x = x.transpose(-1, -2)
        _, (hn, cn) = self.lstm(x)
        x = hn[-1]
        x = self.norm(x)
        return x

class LSTMCore(torch.nn.Module):

    def __init__(self, band_amount = 4, category_num = 3, band_feature_dim = 125):
        super().__init__()
        channels = 14
        self.streams = torch.nn.ModuleList()
        for _ in range(band_amount):
            stream_module = torch.nn.Sequential(
                    LSTMStream(channels, band_feature_dim),
                    torch.nn.Linear(band_feature_dim, 3)
                    )
            self.streams.append(stream_module)
        self.linear = torch.nn.Linear(band_amount * 3, category_num)

    def forward(self, x):
        '''
        x: [bs, band, channel, timestep]
        '''
        feature_list = []
        for i,stream in enumerate(self.streams):
            feature = stream(x[:, i])
            feature_list.append(feature)
        total_feature = torch.cat(feature_list, dim=-1)
        y = self.linear(total_feature)
        return y

class LSTMBaseline(LSMBase):

    def __init__(self, band_amount=4, category_num=3, band_feature_dim = 125):
        super().__init__()
        self.fake_reservoir = torch.nn.Linear(2,2)
        self.LSTM_core = LSTMCore(band_amount, category_num, band_feature_dim)
        self.register(self.fake_reservoir, self.LSTM_core)

    def forward(self, x):
        y = self.LSTM_core(x)
        return y

# Following: GRU Baseline

class GRUStream(torch.nn.Module):

    def __init__(self, original_channels, out_dim):
        super().__init__()

        self.gru = torch.nn.GRU(
                input_size = original_channels,
                hidden_size = out_dim,
                num_layers = 1,
                batch_first = True
                )
        self.norm = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        '''
        x: [batch_size, channels, timesteps]
        return: [batch_size, output_dim]
        '''
        x = x.transpose(-1, -2)
        _, hn = self.gru(x)
        x = hn[-1]
        x = self.norm(x)
        return x

class GRUCore(torch.nn.Module):

    def __init__(self, band_amount = 4, category_num = 3, band_feature_dim = 125):
        super().__init__()
        channels = 14
        self.streams = torch.nn.ModuleList()
        for _ in range(band_amount):
            stream_module = torch.nn.Sequential(
                    GRUStream(channels, band_feature_dim),
                    torch.nn.Linear(band_feature_dim, 3)
                    )
            self.streams.append(stream_module)
        self.linear = torch.nn.Linear(band_amount * 3, category_num)

    def forward(self, x):
        '''
        x: [bs, band, channel, timestep]
        '''
        feature_list = []
        for i,stream in enumerate(self.streams):
            feature = stream(x[:, i])
            feature_list.append(feature)
        total_feature = torch.cat(feature_list, dim=-1)
        y = self.linear(total_feature)
        return y

class GRUBaseline(LSMBase):

    def __init__(self, band_amount=4, category_num=3, band_feature_dim = 125):
        super().__init__()
        self.fake_reservoir = torch.nn.Linear(2,2)
        self.GRU_core = GRUCore(band_amount, category_num, band_feature_dim)
        self.register(self.fake_reservoir, self.GRU_core)

    def forward(self, x):
        y = self.GRU_core(x)
        return y

# Following: EEGNet Baseline

class EEGNetCore(torch.nn.Module):

    def __init__(self, window_length: int, band_amount = 4, category_num = 3):
        super().__init__()
        channels = 14
        self.streams = torch.nn.ModuleList()
        for _ in range(band_amount):
            stream_module = braindecode.models.EEGNet(
                    n_chans = channels,
                    n_outputs = 3,
                    n_times = window_length
                    )
            self.streams.append(stream_module)
        self.linear = torch.nn.Linear(band_amount * 3, category_num)

    def forward(self, x):
        '''
        x: [bs, band, channel, timestep]
        '''
        feature_list = []
        for i,stream in enumerate(self.streams):
            feature = stream(x[:, i])
            feature_list.append(feature)
        total_feature = torch.cat(feature_list, dim=-1)
        y = self.linear(total_feature)
        return y

class EEGNetBaseline(LSMBase):

    def __init__(self, window_length, band_amount=4, category_num=3):
        super().__init__()
        self.fake_reservoir = torch.nn.Linear(2,2)
        self.EEGNet_core = EEGNetCore(window_length, band_amount, category_num)
        self.register(self.fake_reservoir, self.EEGNet_core)

    def forward(self, x):
        y = self.EEGNet_core(x)
        return y
