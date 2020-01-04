import torch

dataPath = '../../../Data/minist'
inputDim = 1
outputDim = 1
epoch = 100000
learnR = 0.02
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
