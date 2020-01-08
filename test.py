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

class LogisticR(nn.Module):
    def __init__(self):
        super(LogisticR, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        #return self.softmax(self.fc(x))
        return self.fc(x)

model = LogisticR()
model = model.to(device)
model.load_state_dict(torch.load('./modelStateDict'))

output = model(feaTh)
prediction = torch.argmax(output, dim= 1)

# Move the tensor from gpu to cpu
df = pd.DataFrame(data= {'ImageId': np.arange(1, dataR.shape[0] + 1, step= 1),
                                           'Label': prediction.cpu().numpy()})

df.to_csv('../../../Data/minist/sub.csv', index= False)
print ('Finished test!!')