# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 12

This folder consists of Assignment-12 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-12
Move your S10 assignment to Lightning first and then to Spaces such that:
-- You have retrained your model on Lightning
-- You are using Gradio
-- Your spaces app has these features:
- ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
- ask whether he/she wants to view misclassified images, and how many
- allow users to upload new images, as well as provide 10 example images
- ask how many top classes are to be shown (make sure the user cannot enter more than 10)
- Add the full details on what your App is doing to Spaces README 

The assignment consists of 3 files and 1 folder
* custom_resnet.py - This file is the custom resnet model used in Assignment - 10, migrated to pytorch
* utils.py - A utility of commonly used functions. Also consists of helper functions to train and test the model. 
* S12.ipnyb - Jupyter notebook that consists of the assignment - training and testing CIFAR-10 data with our custom resnet model.


The custom resnet model, has the following model architecture

-----------------------------------------------------------
    #    | Name                 | Type               | Params
-------------------------------------------------------------
    0       | loss_criteria        | CrossEntropyLoss   | 0     
    1       | accuracy             | MulticlassAccuracy | 0     
    2       | convblockPreparation | Sequential         | 1.9 K 
    3       | convblockL1X1        | Sequential         | 74.0 K
    4       | convblockL1R1        | Sequential         | 295 K 
    5       | convblockL2X1        | Sequential         | 295 K 
    6       | convblockL3X1        | Sequential         | 1.2 M 
    7       | convblockL3R1        | Sequential         | 4.7 M 
    8       | FinalBlock           | Sequential         | 0     
    9       | FC                   | Sequential         | 5.1 K 
    10      | dropout              | Dropout            | 0     
-------------------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.293    Total estimated model params size (MB)

### RESULTS:

A test accuracy of 91.93% is seen  with cifar dataset

Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████| 20/20 [00:03<00:00,  5.72it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   Runningstage.testing    ┃                           ┃
┃          metric           ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│          vl_acc           │    0.9193000197410583     │
│          vl_loss          │    0.24440039694309235    │
└───────────────────────────┴───────────────────────────┘
[{'vl_loss': 0.24440039694309235, 'vl_acc': 0.9193000197410583}]

### Hugging Face - Spaces app:
https://huggingface.co/spaces/ToletiSri/TSAI_S12

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

#### Get misclassified images
```http
  getMiscassifications(model, data_loader, count)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model |
| data_loader |torch.utils.data.dataloader.DataLoader| Required: Loader consisting of test data |
| count | int | Optional:  Number of misclassified images to obtain. Default is 10 |

#### Get Gradcam image
```http
  getGradCamImage(model, norm_image, means, stds)
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model |
| norm_image |torch.tensor| Required: Normalised image |
| means | list | Required: List of means of the  image in each channel|
| stds | liat | Required: List of Standard Deviations of the  image in each channel |

#### Plot misclassified gradcam images
```http
  displayGradCamForInvalidData(model, data_loader,means, stds, count = 10)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| model | torch.nn.Module | Required: PyTorch model |
| data_loader |torch.utils.data.dataloader.DataLoader| Required: Loader consisting of test data |
| means | list | Required: List of means of the  image in each channel|
| stds | liat | Required: List of Standard Deviations of the  image in each channel |
| count | int | Optional:  Number of misclassified images to obtain. Default is 10 |
