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

dataR = pd.read_csv(R.dataPath)

tarNp = dataR.label.values
feaNp = dataR.loc[:, dataR.columns != 'label'].values / 255

tarTh = torch.from_numpy(tarNp)
feaTh = torch.from_numpy(feaNp).type(torch.float)

batchSize = 100
epoch  = 10
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tarTh = tarTh.to(dev)
feaTh = feaTh.to(dev)

trainDs = torch.utils.data.TensorDataset(feaTh, tarTh)
trainLd = torch.utils.data.DataLoader(trainDs, batch_size= batchSize, shuffle= False)

#plt.imshow(feaNp[9, :].reshape((28, 28)))
#plt.title(tarNp[9])
#plt.axis('off')
#plt.show()

class LogisticR(nn.Module):
    def __init__(self):
        super(LogisticR, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        #return self.softmax(self.fc(x))
        return self.fc(x)

learnR = 0.02
model = LogisticR()
model = model.to(dev)
lossF = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr= learnR)

loss1d = []
fig, axe = plt.subplots(nrows= 1, ncols= 1)
for i in range(epoch):
    print ('Training epoch: ', i + 1)
    for idx, (fea, label) in enumerate(trainLd):

        opt.zero_grad()
        output = model(fea.reshape(-1, 28 * 28))

        loss = lossF(output, label)
        loss1d.append(loss)
        loss.backward()
        opt.step()
    axe.plot(loss1d, label= 'epoch {}'.format(i + 1))

torch.save(model.state_dict(), './modelStateDict')
print ('Saved model state dict')

axe.set_xlabel('Iteration')
axe.set_ylabel('LossVal')
axe.legend()
plt.show()


