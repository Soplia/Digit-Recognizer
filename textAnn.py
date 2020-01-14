import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import perf_counter
import os
import Resource as R

dataR = pd.read_csv('../../../Data/minist/test.csv')
feaTh = torch.from_numpy(dataR.loc[:, dataR.columns != 'label'].values / 255)
feaTh = feaTh.type(torch.float)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
feaTh = feaTh.to(device)

class ANN(nn.Module):
    def __init__(self, intputDim, hiddenDim, outputDim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(intputDim, hiddenDim)
        self.relu = nn.ReLU()
        self.h1 = nn.Linear(hiddenDim, hiddenDim)
        self.tanh = nn.Tanh()
        self.h2 = nn.Linear(hiddenDim, hiddenDim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hiddenDim, outputDim)
    def forward(self, x):
        out1 = self.relu(self.fc1(x))
        out2 = self.tanh(self.h1(out1))
        out3 = self.elu(self.h2(out2))
        out4 = self.fc2(out3)
        return out4

model = ANN(28 * 28, 150, 10)
model = model.to(device)
model.load_state_dict(torch.load('./modelStateDictAnn'))

output = model(feaTh)
prediction = torch.argmax(output, dim= 1)

# Move the tensor from gpu to cpu
df = pd.DataFrame(data= {'ImageId': np.arange(1, dataR.shape[0] + 1, step= 1),
                                           'Label': prediction.cpu().numpy()})

df.to_csv('../../../Data/minist/subann.csv', index= False)
print ('Finished test!!')
