from utils import torch,nn,F,optim,datasets,transforms
import utils
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
import os
from torch.utils.data import DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label
    

dropout_value = 0.1
class LitCustomResNet(LightningModule):
    def __init__(self,loss_criteria, learning_rate=1e-1):
        super().__init__()
        self.loss_criteria = loss_criteria
        self.learning_rate = learning_rate
        #self.num_classes = 10 #needed to calculate accuracy, to determine if single class or multiclass
        self.accuracy = Accuracy(task='multiclass',num_classes=10)
        
        # Preparation Layer
        self.convblockPreparation = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 32x32x64, RF = 3
        
        
        # Layer 1
        # Layer1, X = 
        self.convblockL1X1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 5
            nn.MaxPool2d(2, 2),            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 16x16x128; RF = 6
        
        #Layer1, res1 = 
        self.convblockL1R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 10           
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # output_size = 16x16x128; RF = 10
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),           
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )  # output_size = 16x16x128; RF = 14
            
        
        # Layer 2       
        self.convblockL2X1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 18
            nn.MaxPool2d(2, 2), #RF = 20            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 8x8x256; RF = 20
            
        
        # Layer 3
        # Layer3, X = 
        self.convblockL3X1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False), #RF = 28
            nn.MaxPool2d(2, 2),  #RF = 32           
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ) # output_size = 4x4x512; RF = 32
        #Layer3, res1 = 
        self.convblockL3R1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),  #RF = 50          
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # output_size = 4x4x512; RF = 50
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation = 1, bias=False),   #RF = 66            
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.Dropout(dropout_value)
        )  # output_size = 4x4x512; RF = 66
            

        #LayerFinal 
        self.FinalBlock = nn.Sequential(
             nn.MaxPool2d(4, 4),  ## output_size = 1x1x512; RF = 74             
        )
        
        #LayerFC
        self.FC = nn.Sequential(
             nn.Linear(512, 10),  ## output_size = 1x1x512; RF = 74             
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #print('shape of x before preparation = ', x.shape)
        #Prep
        x = self.convblockPreparation(x)
        
        #print('Shape after preparation = ',x.shape)
        #L1
        x = self.convblockL1X1(x)
        #print('Shape after L1 X1 = ',x.shape)
        x = x + self.convblockL1R1(x)
        #print('Shape after L1 R1 = ',x.shape)
        
        #L2
        x = self.convblockL2X1(x)
        #print('Shape after L2 = ',x.shape)

        #L3
        x = self.convblockL3X1(x)
        #print('Shape after L3 X1 = ',x.shape)
        x = x + self.convblockL3R1(x)
        #print('Shape after L3 R1 = ',x.shape)
      
        #Final       
        x = self.FinalBlock(x)
        #print('Shape after final = ',x.shape)

        x = x.view(-1, 512)
        #print('Shape after view = ',x.shape)
        
        x = self.FC(x)
        #print('Shape after FC = ',x.shape)
        return x.view(-1, 10)
       
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Predict
        pred = self.forward(x)
        # Calculate loss
        loss = self.loss_criteria(pred, y)    
        preds = torch.argmax(pred, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Predict
        pred = self.forward(x)
        # Calculate loss
        loss = self.loss_criteria(pred, y)        
    
        preds = torch.argmax(pred, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download       
        Cifar10SearchDataset(root='./data', train=True, download=True)
        Cifar10SearchDataset(root='./data', train=False, download=True)

    def setup(self, stage=None):        
        
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]
        BATCH_SIZE = 512

        train_transforms = A.Compose(
            [           
                A.augmentations.transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1, always_apply=False, p = 0.5),       
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
        
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means,  always_apply=True),       
                ToTensorV2(),
            ]
        )

        test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = Cifar10SearchDataset(root='./data', train=True, download=True, transform=train_transforms)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])
            
            def imshow(img):
                import matplotlib.pyplot as plt
                import numpy as np
                img = img / 2 + 0.5     # unnormalize
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.show()

            train_loader = self.train_dataloader()
            # get some random training images
            dataiter = iter(train_loader)
            images, labels = next(dataiter)
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            import torchvision
            # show images
            imshow(torchvision.utils.make_grid(images[:4]))
            # print labels
            print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = Cifar10SearchDataset(root='./data', train=False, download=True, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=512, num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=512, num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=512, num_workers=4, persistent_workers=True, pin_memory=True)
    
    
def getModel(loss_criteria, learning_rate=1e-1):
    return LitCustomResNet(loss_criteria, learning_rate).to(utils.getDevice())


    

    
    
     