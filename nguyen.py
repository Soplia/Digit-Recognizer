import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import time

PATH = '../../../Data/minist/'

trainDf = pd.read_csv(f'{PATH}train.csv')
test_df = pd.read_csv(f'{PATH}test.csv')
submit_df = pd.read_csv(f'{PATH}sample_submission.csv')

#testDf = pd.read_csv(f'{PATH}test.csv')
#print (trainDf.head())
num_pixel = len(trainDf.columns) - 1

transform_0 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,0.8)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=30, scale=(1.1,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_4 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=30, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_5 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=10, shear=45),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class DataFrame_to_Dataset(Dataset):
    
    def __init__(self, df, transform=transform_0):

        # Get features and labels
        if len(df.columns) == num_pixel:
            # Test dataset
            self.features = df.values.reshape((-1,28,28)).astype(np.uint8) # .astype(np.uint8) for ToPILImage transformer
            self.labels = None
        else:
            # Train dataset
            self.features = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)
            self.labels = torch.from_numpy(df.label.values)
        
        # Transformer
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        if self.labels is not None:
            return self.transform(self.features[index]), self.labels[index]
        else:
            return self.transform(self.features[index])

def create_dataloaders(seed, test_size=0.1, df=trainDf, batch_size=32):
    # Create training set and validation set
    train_data, valid_data = train_test_split(df,
                                              test_size=test_size,
                                              random_state=seed)
    
    # Create Datasets
    train_dataset_0 = DataFrame_to_Dataset(train_data)
    train_dataset_1 = DataFrame_to_Dataset(train_data, transform_1)
    train_dataset_2 = DataFrame_to_Dataset(train_data, transform_2)
    train_dataset_3 = DataFrame_to_Dataset(train_data, transform_3)
    train_dataset_4 = DataFrame_to_Dataset(train_data, transform_4)
    train_dataset_5 = DataFrame_to_Dataset(train_data, transform_5)
    train_dataset = ConcatDataset([train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5])

    valid_dataset = DataFrame_to_Dataset(valid_data)
    
    # Create Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

class Net(nn.Module):
    def __init__(self):
        # Super function. It inherits from nn.Module and we can access everythink in nn.Module
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), # inplace=True helps to save some memory
            
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=14), # compute same padding by hand
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # inplace=True helps to save some memory
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=6), # compute same padding by hand
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128*1*1, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128*1*1)
        x = self.fc(x)
        
        return x

# Get the device
use_cuda = torch.cuda.is_available()

def train(seed, num_epochs):
    
    # Train and valid dataloaders
    print('Creating new dataloaders...')
    train_loader, valid_loader = create_dataloaders(seed=seed)

    # Model
    print('Creating a new model...')
    net = Net()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Move to GPU
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), 
                           lr=0.003, betas=(0.9, 0.999), 
                           eps=1e-08, weight_decay=0, 
                           amsgrad=False)

    # Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print('Training the model...')
    for epoch in range(num_epochs):
        
        net.train() # Training mode -> Turn on dropout
        t0 = time.time()
        training_loss = 0.0
        num_samples = 0
        
        for features, labels in train_loader:
            
            # Move data to GPU
            if use_cuda:
                features = features.cuda()
                labels = labels.cuda()
                
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update loss
            training_loss += loss.item()
            num_samples += len(features)
            
        # Compute accuracy on validation set
        net.eval() # Evaluation mode -> Turn off dropout
        correct = 0
        total = 0
        with torch.no_grad():
            for valid_features, valid_labels in valid_loader:
                
                # Move data to GPU
                if use_cuda:
                    valid_features = valid_features.cuda()
                    valid_labels = valid_labels.cuda()

                outputs = net(valid_features)
                _, predicted = torch.max(outputs, 1)
                total += valid_labels.size(0)
                correct += (predicted == valid_labels).sum().item()
        
        scheduler.step()
        
        print('[model %d, epoch %d, time: %.3f seconds] train_loss: %.5f, val_acc: %4f %%' %
              (seed + 1, epoch + 1, time.time() - t0, training_loss/num_samples, 100 * correct / total))
    
    # Prediction
    net.eval() # Evaluation mode -> Turn off dropout
    test_pred = torch.LongTensor()
    
    if use_cuda:
        test_pred = test_pred.cuda()
        
    with torch.no_grad(): # Turn off gradients for prediction, saves memory and computations
        for features in test_loader:

            if use_cuda:
                features = features.cuda()

            # Get the softmax probabilities
            outputs = net(features)
            # Get the prediction of the batch
            _, predicted = torch.max(outputs, 1)
            # Concatenate the prediction
            test_pred = torch.cat((test_pred, predicted), dim=0)
    
    model_name = 'model_' + str(seed + 1)
    ensemble_df[model_name] = test_pred.cpu().numpy()
    print('Prediction Saved! \n')

    # Create test_loader
test_dataset = DataFrame_to_Dataset(test_df)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

ensemble_df = submit_df.copy()

num_models = 1
num_epochs = 6

for seed in range(num_models):
    train(seed, num_epochs)
#torch.save(net.state_dict(), './modelStateDictNguyen')
# Final prediction
final_pred = ensemble_df.iloc[:,2:].mode(axis=1).iloc[:,0]
submit_df.Label = final_pred.astype(int)
submit_df.head()

# Create a submission file
submit_df.to_csv('nguyen.csv', index=False)