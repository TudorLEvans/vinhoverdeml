import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

# ++++++++ LOAD DATA WITH PANDAS ++++++++ #

wine = pd.read_csv('winequality-red.csv',';')

def normalise(wine):
    return (wine-wine.min())/(wine.max()-wine.min())

train_data, train_result = wine.iloc[0:1200,0:11], wine.iloc[0:1200,11]
test_data, test_result = wine.iloc[1200:,0:11], wine.iloc[1200:,11]

train_data = normalise(train_data)
test_data = normalise(test_data)

def map_to_array(input):
    zeros = np.zeros(10)
    zeros[input] = 1
    return zeros

def array_for(x):
    return np.array([map_to_array(xi) for xi in x])

train_result = array_for(train_result)
test_result = array_for(test_result)

# +++++++++ GENERATE TENSORS WHICH WILL AUTO LOOP THROUGH MINI-BATCH +++++++++++ #

train_tensor, train_result_tensor, test_tensor, test_result_tensor = map(
    torch.tensor, (train_data.values, train_result, test_data.values, test_result)
)

train_ds = TensorDataset(train_tensor, train_result_tensor)
train_dl = DataLoader(train_ds, batch_size=40)

# +++++++++ DEFINE STRUCTURE OF NET ++++++++++ #

loss_func = F.mse_loss

class wine_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(11,15)
        self.lin2 = nn.Linear(15,10)

    def forward(self,xd):
        xd = torch.sigmoid(self.lin1(xd))
        xd = torch.sigmoid(self.lin2(xd))
        return xd

def get_model():
    model = wine_model().double()
    return model, optim.SGD(model.parameters(), lr=1)

# +++++++++ DEFINE TRAINING PARAMS AND LOOP +++++++++++ #

model, opt = get_model()

def fit(epochs):
    correct = np.zeros(epochs)
    for epoch in range(epochs):
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred,yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        test_output = model(test_tensor)
        for j in range(test_output.shape[0]):
            values, indices = test_output[j].max(0)
            for i in range(test_output.shape[1]):
                if (i == indices):
                    test_output[j,i] = 1
                else:
                    test_output[j,i] = 0
                if (test_output[j,i] == 1 and test_result_tensor[j,i] == 1):
                    correct[epoch] += 1
    plt.plot(100*correct/399)
    plt.xlabel('epoch')
    plt.ylabel('percent accuracy')
    plt.show()

fit(80)


    
