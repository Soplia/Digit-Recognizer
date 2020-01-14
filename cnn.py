import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import pandas as pd
import numpy as np
import Resource as R
import matplotlib.pyplot as plt

train = pd.read_csv(R.dataPath)

nTrain = len(train)
nPrixel = len(train.columns) - 1
nClass = len(set(train['label']))
print ('Len of  train dataset is {},\ntotally {} columns, \n{} classes '.format(nTrain, nPrixel, nClass))

randNum = np.random.randint(nTrain, size= 8)
print (*list(train.iloc[randNum, 0]), sep = ', ')
grid = make_grid(torch.tensor(train.iloc[randNum, 1:].values / 255.).reshape(-1, 28, 28).unsqueeze(1))
plt.imshow(grid.numpy().transpose( 1, 2, 0))
plt.axis('off')
plt.show()

plt.bar(train['label'].value_counts().index, train['label'].value_counts())
plt.xticks(np.arange(start= 0, stop= nClass, step= 1))
plt.show()