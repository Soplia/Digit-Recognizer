import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import os
import Resource as R
print (os.listdir(R.dataPath))
print (torch.cuda.is_available())

class LinearR(nn.Module):
    def __init__(self, input, output):
        super(LinearR, self).__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        return self.linear(x)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinearR(R.inputDim, R.outputDim)
model = model.to(dev)
lossF = nn.MSELoss()
op = torch.optim.SGD(model.parameters(), R.learnR)


car_prices_array = [3,4,5,6,7,8,9]
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
carPrice = torch.tensor(car_prices_array, dtype= torch.float32, device= dev)
carNum = torch.tensor(number_of_car_sell_array, dtype= torch.float32, device= dev)

#carPrice = torch.tensor(car_prices_array, dtype= torch.float32)
#carNum = torch.tensor(number_of_car_sell_array, dtype= torch.float32)

lossL = []
print ('Start')
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#t1_start = perf_counter()  
for i in range(R.epoch):
    op.zero_grad()
    prediction = model(carPrice.view(-1, 1))
    lossC = lossF(prediction, carNum.view(-1, 1))
    lossC.backward()
    op.step()
    lossL.append(lossC)
    #if i % 20 == 0:
    #    print ('Epoch {} : loss is {}...'.format(i + 1, lossC))

#t1_stop = perf_counter()  
#print("Elapsed time during the whole program in seconds:", 
#                                        t1_stop - t1_start)
end.record()
torch.cuda.synchronize()
print("Elapsed time during the whole program in seconds:", start.elapsed_time(end))

plt.plot(lossL)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()




