import os
import torch
import numpy as np
import scipy as sp
import pytorch_forecasting
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
from model_utils import Gamma_regression, Regression_Dataset

if torch.cuda.is_available():
    device = 'cuda:{}'.format(torch.cuda.current_device())
else:
    device = 'cpu'

### generate a gamma distributed dataset
seed = 200
feature_dim = 10
x, y = make_regression(n_samples=3000, n_features=feature_dim, random_state=seed)
y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
y = np.abs(y)
y = sp.special.gdtrc(1, 0.3, y)

### initialize the model
model = Gamma_regression(feature_dim=feature_dim, dense_units=[15, 20, 25, 30])
# model = model.to(device)
model = model.cuda(2)
print(summary(model=model, input_size=(10,), verbose=0))

a = torch.Tensor(3, 10)
torch.nn.init.normal_(a)

### initialize the dataloader
regression_data = Regression_Dataset(x=x, y=y)
batch_size = 64
num_workers = 16
n_epochs = 1500
train_dataloader = torch.utils.data.DataLoader(regression_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=num_workers
                                               )

### define the loss function and optimizer
loss_function = pytorch_forecasting.MAPE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(n_epochs):
    step = 0
    train_loss = 0.0
    with tqdm(train_dataloader, desc='Training', unit='batch') as tepoch:
        for data in tepoch:
            tepoch.set_description('epoch: {}'.format(epoch + 1))
            temp_out = model(torch.tensor(data[0].to(model.device), dtype=torch.float32))
            loss = loss_function(temp_out,
                                 torch.tensor(data[1].to(model.device), dtype=torch.float32).unsqueeze(dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1
            tepoch.set_postfix(loss=train_loss / step)
