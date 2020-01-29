import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import pandas as pd
import numpy as np
import Resource as R
import matplotlib.pyplot as plt

testDf = pd.read_csv('../../../Data/minist/test.csv')
testTh = torch.from_numpy(testDf.values / 255).reshape(-1, 1, 28, 28).type(torch.float)
testTh = testTh.to(R.device)

testDs = torch.utils.data.TensorDataset(testTh)
bathSize = 1000
testLd = torch.utils.data.DataLoader(dataset= testDs, batch_size=bathSize, shuffle= False)

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
model.load_state_dict(torch.load('./modelStateDictCNN'))


def prediction(dataLd):
    model.eval()
    predList = torch.zeros((1, 1))
    for fea in testLd:
        outputs = model(fea[0])
        pred  =np.argmax(outputs.detach().numpy(), axis= 1)
        predList = np.hstack((predList, pred.reshape(1, -1)))
    
    df =pd.DataFrame({'ImageId': np.arange(1, testDf.shape[0] + 1, step= 1)
                                , 'Label': predList[0, 1:].astype(np.int)})
    df.to_csv('../../../Data/minist/subcnn.csv', index= False)

prediction(testLd)
print ('Finish Test')