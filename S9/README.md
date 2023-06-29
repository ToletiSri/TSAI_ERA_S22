# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 9

This folder consists of Assignment-9 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-9
Train a CIFAR-10 dataset which satisfies the following conditions:
- accuracy of 85% and using less than 200k parameters
- The network should have the following architecture:
C1-C2-C3-C4O (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- No limit on epochs
- Total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use albumentation library and apply:
-- horizontal flip
-- shiftScaleRotate
-- coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px,min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

Well, I have not used Maxpooling or strided convolutions. I could get the results using DILATED CONVOLUTIONS ONLY (**Expecting the bonus 200 points!!**) 
But, I ran out of time to implement albumentations :( Crazy data load errors implementing albumentations :(

The assignment consists of 3 files and 1 folder
* models.py - This file consists of our custom CNN-model.
* utils.py - A utility of commonly used functions. Also consists of helper functions to train and test the model. 
* S9.ipnyb - Jupyter notebook that consists of the assignment - training and testing CIFAR-10 data with our custom CNN model.


The custom model has the following model parameters:

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           9,216
             ReLU-10           [-1, 32, 32, 32]               0
      BatchNorm2d-11           [-1, 32, 32, 32]              64
          Dropout-12           [-1, 32, 32, 32]               0
           Conv2d-13           [-1, 32, 32, 32]           9,216
             ReLU-14           [-1, 32, 32, 32]               0
      BatchNorm2d-15           [-1, 32, 32, 32]              64
          Dropout-16           [-1, 32, 32, 32]               0
           Conv2d-17           [-1, 64, 28, 28]          18,432
             ReLU-18           [-1, 64, 28, 28]               0
      BatchNorm2d-19           [-1, 64, 28, 28]             128
          Dropout-20           [-1, 64, 28, 28]               0
           Conv2d-21           [-1, 32, 28, 28]           2,048
           Conv2d-22           [-1, 32, 28, 28]             320
           Conv2d-23           [-1, 64, 28, 28]           2,112
             ReLU-24           [-1, 64, 28, 28]               0
      BatchNorm2d-25           [-1, 64, 28, 28]             128
          Dropout-26           [-1, 64, 28, 28]               0
           Conv2d-27           [-1, 64, 28, 28]             640
           Conv2d-28           [-1, 64, 28, 28]           4,160
             ReLU-29           [-1, 64, 28, 28]               0
      BatchNorm2d-30           [-1, 64, 28, 28]             128
          Dropout-31           [-1, 64, 28, 28]               0
           Conv2d-32           [-1, 64, 22, 22]          36,864
             ReLU-33           [-1, 64, 22, 22]               0
      BatchNorm2d-34           [-1, 64, 22, 22]             128
          Dropout-35           [-1, 64, 22, 22]               0
           Conv2d-36           [-1, 32, 22, 22]           2,048
           Conv2d-37           [-1, 32, 22, 22]             320
           Conv2d-38           [-1, 64, 22, 22]           2,112
             ReLU-39           [-1, 64, 22, 22]               0
      BatchNorm2d-40           [-1, 64, 22, 22]             128
          Dropout-41           [-1, 64, 22, 22]               0
           Conv2d-42           [-1, 64, 22, 22]             640
           Conv2d-43           [-1, 64, 22, 22]           4,160
             ReLU-44           [-1, 64, 22, 22]               0
      BatchNorm2d-45           [-1, 64, 22, 22]             128
          Dropout-46           [-1, 64, 22, 22]               0
           Conv2d-47           [-1, 64, 14, 14]          36,864
             ReLU-48           [-1, 64, 14, 14]               0
      BatchNorm2d-49           [-1, 64, 14, 14]             128
          Dropout-50           [-1, 64, 14, 14]               0
           Conv2d-51           [-1, 32, 14, 14]           2,048
           Conv2d-52           [-1, 32, 12, 12]           9,216
             ReLU-53           [-1, 32, 12, 12]               0
      BatchNorm2d-54           [-1, 32, 12, 12]              64
          Dropout-55           [-1, 32, 12, 12]               0
           Conv2d-56           [-1, 32, 12, 12]           9,216
             ReLU-57           [-1, 32, 12, 12]               0
      BatchNorm2d-58           [-1, 32, 12, 12]              64
          Dropout-59           [-1, 32, 12, 12]               0
           Conv2d-60             [-1, 32, 4, 4]           9,216
             ReLU-61             [-1, 32, 4, 4]               0
      BatchNorm2d-62             [-1, 32, 4, 4]              64
          Dropout-63             [-1, 32, 4, 4]               0
        AvgPool2d-64             [-1, 32, 1, 1]               0
           Conv2d-65             [-1, 10, 1, 1]             320
================================================================
Total params: 170,592
Trainable params: 170,592
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.40
Params size (MB): 0.65
Estimated Total Size (MB): 14.06
----------------------------------------------------------------


## API Reference - model.py

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

