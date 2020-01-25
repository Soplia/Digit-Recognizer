import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import pandas as pd
import numpy as np
import Resource as R
import matplotlib.pyplot as plt

trainDf = pd.read_csv(R.dataPath)
numClass = len(set(trainDf.label.values))

print ('Train data set shape: ', trainDf.shape)
print ('Number of class: ', numClass)

#randData = np.random.randint(low= 0, high= trainDf.shape[0], size = 8)
#grid = make_grid(torch.tensor(trainDf.iloc[randData, 1:].values.reshape(-1, 28, 28) / 255).unsqueeze(1), 
#                 nrow= 8)

#plt.imshow(grid.numpy().transpose(1, 2, 0))
#print (*list(trainDf.iloc[randData, 0].values), sep= ',')
#plt.show()

#plt.bar(trainDf['label'].value_counts().index, trainDf['label'].value_counts())
#plt.xticks(np.arange(numClass))
#plt.show()

feaTrain = torch.from_numpy(trainDf.iloc[:, trainDf.columns != 'label'].values / 255).type(torch.float).to(R.device)
tarTrain = torch.from_numpy(trainDf.label.values).to(R.device)
#feaTrain = torch.from_numpy(trainDf.loc[:, trainDf.columns != 'label'].values / 255).type(torch.float)
#tarTrain = torch.from_numpy(trainDf.label.values)

batchSize = 1000

trainDs = torch.utils.data.TensorDataset(feaTrain, tarTrain)
trainLd = torch.utils.data.DataLoader(trainDs, batch_size= batchSize, shuffle= False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
            )
        for m in self.feature.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                # 原来是这样子的
                #nn.init.xavier_uniform(m.weight)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        out = self.classifier(x)
        return out

model = CNN()
model = model.to(R.device)
opt = torch.optim.Adam(model.parameters(), lr= 0.02)
lossF = nn.CrossEntropyLoss()
lossF = lossF.to(R.device)
# 用于调整学习速率
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size= 7, gamma= 0.1)

def train(epoch):
    #model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. 
    model.train()
    for batch_idx, (image, target) in enumerate(trainLd):
        data = image.reshape(batchSize, 1, 28, 28)
        opt.zero_grad()
        output = model(data)
        loss = lossF(output, target)
        loss.backward()
        opt.step()
        # In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.
        exp_lr_scheduler.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(trainLd.dataset),
                100. * (batch_idx + 1) / len(trainLd), loss.item()))

def evaluate(data_loader):
    #model.eval() or model.train(mode=False) to tell that you are testing.
    model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        data = data.reshape(batchSize, 1, 28, 28)
        output = model(data)
        loss += F.cross_entropy(output, target, reduction= 'sum').item()

        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred == target)

    loss /= len(data_loader.dataset)
        
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

n_epochs = 20

for epoch in range(n_epochs):
    train(epoch)
    evaluate(trainLd)

torch.save(model.state_dict(), './modelStateDictCNN')