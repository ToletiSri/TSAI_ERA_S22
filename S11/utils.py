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
        

def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
        
def plotImageCIFAR10(loader):
    import torchvision
    # get some random  images
    dataiter = iter(loader)
    images, labels = next(dataiter)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')   
    # show images
    imshow(torchvision.utils.make_grid(images[:4]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        
def getTrainTransforms_CropRotate(centerCrop, resize, randomRotate,mean,std_dev):    
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(centerCrop), ], p=0.1),
        transforms.Resize((resize, resize)),
        transforms.RandomRotation((-randomRotate, randomRotate), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,)),
        ])
    return train_transforms

def getTrainTransforms(mean,std_dev):    
    # Train data transformations
    train_transforms = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,)),
        ])
    return train_transforms

def getTrainTransformWithCutoutForCIFAR(means,stds):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # Train data transformations
    train_transforms = A.Compose(
        [           
        A.augmentations.transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1, always_apply=False, p = 0.5),       
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),        
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means, always_apply=True),       
        ToTensorV2(),
        ]
    )

    return train_transforms

def getTestTransforms(mean,std_dev):
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std_dev,))
        ])
    return test_transforms

def getTestTransformsForCIFAR(means,stds):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # Test data transformations
    test_transforms = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]
    )
    return test_transforms

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label
    

def getDataWithAlbumentations(root_path, trainData=True,download=True, transforms=''):
    data = Cifar10SearchDataset(root=root_path, train=trainData,
                                        download=download, transform=transforms)
    return data

def getDataLoader(SEED, shuffle, batch_size, num_workers, pin_memory,dataset):
    # For reproducibility
    torch.manual_seed(SEED)     

    if isCUDAAvailable():
        torch.cuda.manual_seed(SEED)
        
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if isCUDAAvailable() else dict(shuffle=shuffle, batch_size=64)
    
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)    
    return data_loader



def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, train_loader, optimizer, criterion, scheduler):
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
        scheduler.step()

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
            test_loss += criterion(output, target).item()  # sum up batch loss

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
    

def getMiscassifications(model, data_loader, count=10):
    import numpy as np
    device = getDevice()
    dataiter = iter(data_loader)
    with torch.no_grad():
        data, target = next(dataiter)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        idxs_mask = (pred == target).squeeze().to('cpu')
        incorrect_idx = np.asarray(np.where(idxs_mask == False)).flatten()[:count]
        
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
        misclassfied_images = data[incorrect_idx]
        misclassfied_label_code = np.asarray(pred[incorrect_idx].to('cpu')).flatten()
        actual_label_code =  np.asarray(target[incorrect_idx].to('cpu')).flatten()
        
        misclassfied_labels = []
        actual_labels = []
        
        for x in range(count):
            misclassfied_labels.append(classes[misclassfied_label_code[x]])
            actual_labels.append(classes[actual_label_code[x]])
            
        return misclassfied_images, misclassfied_labels, actual_labels 
    
def getGradCamImage(model, norm_image, means, stds):
    from gradcam import GradCAM 
    from gradcam.utils import visualize_cam 
    
    norm_image = norm_image.to(getDevice())
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/stds[0], 1/stds[1], 1/stds[2] ]),
                                transforms.Normalize(mean = [ -1*means[0], -1*means[1], -1*means[2] ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    image = invTrans(norm_image)
    model.eval()
    gradcam = GradCAM.from_config(model_type='resnet', arch=model, layer_name='layer3')

    mask, logit = gradcam(norm_image[None])
    # make heatmap from mask and synthesize saliency map using heatmap and img
    heatmap, cam_result = visualize_cam(mask, image)
    return cam_result

def displayGradCamForInvalidData(model, data_loader,means, stds, count = 10):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    
    misclassfied_images, misclassfied_labels,actual_labels = getMiscassifications(model, data_loader, count)
    grad_images = []
    for x in range(count):
        grad_images.append(getGradCamImage(model, misclassfied_images[x], means, stds ))      
   
 
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 5, constrained_layout=True)   
    
    for i in range(2):
        for j in range(math.floor(count/2)):
            # Modify top-left plot
            index = i*math.floor(count/2) + j 
            npimg = grad_images[index].numpy()
            axes[i,j].set_title("Pred = " + misclassfied_labels[index] + "\n actual = " + actual_labels[index], fontsize = 10)
            axes[i,j].imshow(np.transpose(npimg, (1, 2, 0)))


def suggestMaxLR(model, optimizer, criterion, train_loader):
    from torch_lr_finder import LRFinder
    device = getDevice()
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

