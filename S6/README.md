# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 6

This folder consists of Assignment-6 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-6 
Make use the following concepts discussed until Session 6 in creating a CNN model
- How many layers,
- MaxPooling,
- 1x1 Convolutions,
- 3x3 Convolutions,
- Receptive Field,
- SoftMax,
- Learning Rate,
- Kernels and how do we decide the number of kernels?
- Batch Normalization,
- Image Normalization,
- Position of MaxPooling,
- Concept of Transition Layers,
- Position of Transition Layer,
- DropOut
- When do we introduce DropOut, or when do we know we have some overfitting
- The distance of MaxPooling from Prediction,
- The distance of Batch Normalization from Prediction,
- When do we stop convolutions and go ahead with a larger kernel or some other alternative 
- How do we know our network is not going well, comparatively, very early
- Batch Size, and Effects of batch size

The assignment consists of 3 files and 1 folder
* models.py - This file consists of our custom CNN-model.
* utils.py - A utility of commonly used functions. Also consists of helper functions to train and test the model. 
* S6.ipnyb - Jupyter notebook that consists of the assignment - training and testing MNIST data with our custom CNN model.
*  S6 - Assignment QnA - Folder consisting of an example that implements backpropogation from scratch

The custom model has the following model parameters:

----------------------------------------------------------------
        Layer (type)    |           Output Shape    |     Param #
----------------------------------------------------------------

            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4           [-1, 16, 28, 28]           1,168
              ReLU-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
           Dropout-7           [-1, 16, 28, 28]               0
         MaxPool2d-8           [-1, 16, 14, 14]               0
            Conv2d-9            [-1, 8, 14, 14]             136
      BatchNorm2d-10            [-1, 8, 14, 14]              16
           Conv2d-11           [-1, 16, 14, 14]           1,168
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
           Conv2d-14           [-1, 32, 14, 14]           4,640
             ReLU-15           [-1, 32, 14, 14]               0
      BatchNorm2d-16           [-1, 32, 14, 14]              64
          Dropout-17           [-1, 32, 14, 14]               0
        MaxPool2d-18             [-1, 32, 7, 7]               0
           Conv2d-19             [-1, 16, 7, 7]             528
      BatchNorm2d-20             [-1, 16, 7, 7]              32
           Conv2d-21             [-1, 32, 7, 7]           4,640
      BatchNorm2d-22             [-1, 32, 7, 7]              64
             ReLU-23             [-1, 32, 7, 7]               0
          Dropout-24             [-1, 32, 7, 7]               0
           Conv2d-25             [-1, 16, 7, 7]           4,624
      BatchNorm2d-26             [-1, 16, 7, 7]              32
           Conv2d-27             [-1, 10, 7, 7]             170
      BatchNorm2d-28             [-1, 10, 7, 7]              20
        AvgPool2d-29             [-1, 10, 1, 1]               0
================================================================
Total params: 17,462
Trainable params: 17,462
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.93
Params size (MB): 0.07
Estimated Total Size (MB): 1.00
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
