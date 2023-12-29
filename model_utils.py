import torch
from gammainc_activation import Gammainac_activation
from typing import List


class Gamma_regression(torch.nn.Module):
    def __init__(self, feature_dim: int, dense_units: List[int]):
        super(Gamma_regression, self).__init__()
        total_dense_units = [feature_dim] + dense_units
        self.dense_layers = torch.nn.Sequential()
        for i in range(len(total_dense_units) - 1):
            self.dense_layers.append(torch.nn.BatchNorm1d(num_features=total_dense_units[i]))
            self.dense_layers.append(torch.nn.Linear(in_features=total_dense_units[i],
                                                     out_features=total_dense_units[i + 1]
                                                     ))
        self.dense_layers.append(torch.nn.BatchNorm1d(num_features=total_dense_units[-1]))
        self.dense_layers.append(torch.nn.Linear(in_features=total_dense_units[-1],
                                                 out_features=1
                                                 ))
        self.gammainc_activation = Gammainac_activation()

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.gammainc_activation(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class Regression_Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.features = x
        self.targets = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batched_x = self.features[idx, :]
        batched_y = self.targets[idx]
        return batched_x, batched_y
