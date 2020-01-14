import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd  
import Resource as R
import torch.utils.data
import matplotlib.pyplot as plt

dataR = pd.read_csv(R.dataPath)

tarNp = dataR.label.values
feaNp = dataR.loc[:, dataR.columns != 'label'].values / 255

tarTh = torch.from_numpy(tarNp)
feaTh = torch.from_numpy(feaNp).type(torch.float)

batchSize = 100
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tarTh = tarTh.to(dev)
feaTh = feaTh.to(dev)

trainDs = torch.utils.data.TensorDataset(feaTh, tarTh)
trainLd = torch.utils.data.DataLoader(trainDs, batch_size= batchSize, shuffle= False)

#dataR = pd.read_csv(R.dataPath)
#attributes = dataR[:, dataR.columns != 'label'].values / 255
#labels = dataR.label.values
#attTh = torch.from_numpy(attributes)
#labelTh = torch.from_numpy(labels)
#attTh = attTh.to(R.device)
#labelTh = labelTh.to(R.device)

#trainDs = torch.utils.data.Dataset(attTh, labelTh)
#batchSize = 1000
#trainLd = torch.utils.data.DataLoader(trainDs, batch_size= batchSize)

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
model = model.to(R.device)
lossL = nn.CrossEntropyLoss();
optim = torch.optim.SGD(model.parameters(), lr= 0.2);

epoch = 10
lossList = []
accList = []
for i in range(epoch):
    print ('Epoch{}'.format(i + 1))
    for idx, (image, label) in enumerate(trainLd):
        optim.zero_grad()
        outputs = model(image)
        loss = lossL(outputs, label)
        loss.backward()
        optim.step()
        lossList.append(loss)
        predictions = torch.argmax(outputs)
        accList.append(torch.sum(predictions == label) / outputs.shape[0])
torch.save(model.state_dict(), './modelStateDictAnn')

fig, axe = plt.subplots(nrows=1, ncols=2)
axe[0].plot(lossList, color= 'b')
axe[0].plot(lossList, color= 'r', marker= '>', label= 'Loss')
axe[1].plot(accList, color= 'm')
axe[1].plot(accList, color= 'k', marker= '*', label= 'Acc')
axe[0].set_xlabel('Iterations')
axe[0].set_ylabel('LossValue')
axe[1].set_ylabel('AccValue')
axe[0].legend()
axe[1].legend()
plt.show()





