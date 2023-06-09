#Import all modules needed for Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def isCUDAAvailable():
  return torch.cuda.is_available()

def getDevice():
    device = torch.device("cuda" if isCUDAAvailable() else "cpu")
    return device
    
def plotData(loader, count, cmap_code):
    import matplotlib.pyplot as plt
    batch_data, batch_label = next(iter(loader))
    fig = plt.figure()
    
    for i in range(count):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap=cmap_code)
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
        
def getTrainTransforms(centerCrop, resize, randomRotate,mean,std_dev):    
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(centerCrop), ], p=0.1),
        transforms.Resize((resize, resize)),
        transforms.RandomRotation((-randomRotate, randomRotate), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,)),
        ])
    return train_transforms

def getTestTransforms(mean,std_dev):
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,))
        ])
    return test_transforms


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, train_loader, optimizer, criterion):
    from tqdm import tqdm
    model.train()
    pbar = tqdm(train_loader)
    device = getDevice()
    train_loss = 0
    correct = 0
    processed = 0
    train_acc = []
    train_losses = []
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
    
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return train_acc, train_losses

def test(model, test_loader, criterion):
    from tqdm import tqdm
    model.eval()
    device = getDevice()
    
    test_loss = 0
    correct = 0
    test_acc = []
    test_losses = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, test_losses


def printModelSummary(model, inputSize):
    from torchsummary import summary   
    summary(model, input_size=inputSize)
    
def printModelTrainTestAccuracy(train_acc, train_losses, test_acc, test_losses):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
