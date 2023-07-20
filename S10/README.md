# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 9

This folder consists of Assignment-10 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-10
Write a custom ResNet architecture for CIFAR10 that has the following architecture:
###### PrepLayer - 
Conv 3x3 s1, p1) >> BN >> RELU [64k]
###### Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
###### Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
###### Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
###### FC Layer 
###### SoftMax

###### Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation

###### Uses this transform -
RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
Use ADAM, and CrossEntropyLoss
Target Accuracy: 90%

The assignment consists of 3 files and 1 folder
* custom_resnet.py - This file consists of our custom Resnet model, with the architecture stated above
* utils.py - A utility of commonly used functions. Also consists of helper functions to train and test the model. 
* S10.ipnyb - Jupyter notebook that consists of the assignment - training and testing CIFAR-10 data with our custom CNN model.


The custom resnet model, defined as per the restrictions mentioned above, has the following model parameters:

----------------------------------------------------------------
        Layer (type)      |       Output Shape      |   Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
------------------------------------------------------------------
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.07
Estimated Total Size (MB): 33.09
----------------------------------------------------------------

### RESULTS:

By tweaking the albumentations, and oneCycleLR parameters, we have been able to achieve a validation accuracy of 90% by epoch #23

**Epoch 23**
Train: Loss=0.3193 Batch_id=97 Accuracy=88.81: 100%|███████████████████████████████████| 98/98 [00:48<00:00,  2.03it/s]
Test set: Average loss: 0.0006, Accuracy: 9053/10000 (**90.53%**)
LR = [0.00030234014432884896]

**Epoch 24**
Train: Loss=0.3124 Batch_id=97 Accuracy=89.50: 100%|███████████████████████████████████| 98/98 [00:47<00:00,  2.06it/s]
Test set: Average loss: 0.0006, Accuracy: 9063/10000 (**90.63%**)
LR = [1.0570957086617859e-06]

## API Reference - custom_resnet.py

#### Get CNN model

```http
  getModel()
```


## API Reference - utils.py

#### CUDA Availability

```http
  isCUDAAvailable()
```

#### Get PyTorch device

```http
  getDevice()
```

#### Plot Data

```http
  plotData(loader, count, cmap_code)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| loader | torch.utils.data.dataloader.DataLoader | Required: Loader consisting of train/test data to be plotted |
| count | integer | Required: The number of elements from loader to be plotted |
| cmap_code | string | Required: CMAP color code used for plotting |

#### Get Transform (Crop, Resize, Rotate) for train data
```http
  getTrainTransforms(centerCrop, resize, randomRotate,mean,std_dev)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| centerCrop | int | Required: Number of pixels to be cropped at center of the image |
| resize | integer | Required: Number of pixels to resize the cropped image |
| randomRotate | float | Required: Angle to rotate the image during training |
| mean | float | Required: Mean of the input data |
| std_dev | float | Required: Standard deviation of the input data |


#### Get Transform for test data
```http
  getTestTransforms(mean,std_dev)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| mean | float | Required: Mean of the input data |
| std_dev | float | Required: Standard deviation of the input data |


#### Train model
```http
  train(model, train_loader, optimizer, criterion)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model. Use 'model.getModel()' to get an instance of our first PyTorch CNN model|
| train_loader | torch.utils.data.dataloader.DataLoader	 | Required: Loader consisting of train data |
| optimizer | torch.optim  | Required: Optimizer used in training data |
| criterion | function | Required: Fucntion used to calculate the loss during training |


#### Test model
```http
  test(model, test_loader,  criterion)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model. Use 'model.getModel()' to get an instance of our first PyTorch CNN model|
| test_loader | torch.utils.data.dataloader.DataLoader	 | Required: Loader consisting of test data |
| criterion | function | Required: Fucntion used to calculate the loss during test |


#### Plot Accuracy
```http
  printModelTrainTestAccuracy(train_acc, train_losses, test_acc, test_losses)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| train_acc | list | Required: List of training accuracies obtained from 'utils.train()'|
| train_losses |list| Required:  List of train losses obtained from 'utils.train()' |
| test_acc | list | Required:  List of test accuracies obtained from 'utils.test()' |
| test_losses | list | Required: List of test losses obtained from 'utils.test()'|
