import torch

dataPath = '../../../Data/minist/train.csv'
inputDim = 1
outputDim = 1
epoch = 100000
learnR = 0.02
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
