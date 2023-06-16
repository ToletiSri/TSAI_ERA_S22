# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 6

This folder consists of Assignment-7 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-7 sets a target of achieving atleast 99.4% validation accuracy for MNIST data, with a model which has less than 8000 parameters, and, with a maximum of 15 epochs.

###### Folder Structure
Assignment-7 consists of:
- utils.py - Same as utils.py in the previous assignment. This file consists of a bunch of utility functions that are commonly used in model training/validation and plotting of  the results
- /models - This folder consists of the models we have attempted to get the desired results
- Jupyter notebooks - Code1.ipnyb, Code2.ipnyb, Code3.ipnyb etc.. - are the notebooks which consist of the code to train and test each model in the /models folder. Note that each notebook corresponds to one model from the /models folder

## Model Development
Before writing a model, we start with a target. We develop the model keeping this target in mind.
After, running the model, we record the results and analysis for each model. We try to improve with our next model unless our goal is achieved. 

### Model1:
###### Target
A decent receptivity field.
(Start with a basic model. Keep on convoluting, (with o/p channels doubling in enry layer) until a decent receptivity field(21/28) of input image is received.)
###### Result
Model parameters - 6,303,746 
Train Accuracy - 11.24 
Validation Accuracy - 11.35 
##### Analysis
Accuracy is too low. Receptivity field of 21 is not sufficient for image classification in case of MNIST

### Model2:
###### Target
Improve accuracy 
(Trying to improve accuracy by increasing receptivity fiels. Start with 4 output channels. Keep on convoluting, (with o/p channels doubling in every layer) until a decent receptivity field of input image is received. 
Use maxpooling much earlier, so that a decent receptivity field(26/28) is achieved with less channels.)
###### Result
Model parameters - 396,234
Train Accuracy - 99.38
Validation Accuracy - 98.96
##### Analysis
Accuracy improved drastically when receptivity field is increased

### Model3:
###### Target
Reduce the number of model parameters
(64 channels in previous model was heavy on parameters.So try reducing it. 
Keep the convolutions, but, reatining channel size in some layers. Use max pooling)
###### Result
Model parameters - 26,022
Train Accuracy - 98.94
Validation Accuracy - 98.31 
##### Analysis
Accuracy reduced a little compared to previous model, However, this is due to the reduction in number of parameters. 

### Model4:
###### Target
Reduce the number of model parameters further
(Limit max channels to 32, to further reduce model parameters. Keep the convolutions, but, reducing channel size even further. Use max pooling)
###### Result
Model parameters - 7,790
Train Accuracy - 97.88
Validation Accuracy - 97.64
##### Analysis
Although the accuracy reduded further comparitive to the previous model, it's still decent, noting that model parameters reduced from 26k to 7k

### Model5:
###### Target
Reduce Parameters + Improve accuracy
(Try to reduce parameters from 7k to 6k by reducing channel size. Use batch normalisation to achieve better accuracy)
###### Result
Model parameters - 5,438
Train Accuracy - 99.43%
Validation Accuracy - 99.02%
##### Analysis
Best so far! Batch Normalisation Rocks!!

### Model6:
###### Target
Improve accuracy -  explore  
###### Result
Model parameters - 5,438
Train Accuracy - 99.28%
Validation Accuracy - 98.84%%
##### Analysis
I didn't expect droputs to reduce accuracy :(. This is a lesson learnt!

### Model7:
###### Target
Improve accuracy - by increasing receptivity field, using LROnPlateau
(Reduce/retain channel size while adding another layer for receptivity, Increases parameters due to the additional layer. Retain droputs)
###### Result
Model Parameters - 5,918
Train Accuracy - 99.24%
Test accuracy - 99.03%%
##### Analysis
A decent increase in accuracy, but model tends to coverge at a lesser accuracy

### Model8:
###### Target
Improve accuracy - Use Antman and GAP, use LROnPlateau
###### Result
Model Parameters - 4,838
Train Accuracy - 99.32%
Test accuracy - 99.21%%
##### Analysis
Way better in terms of validation accracy. However, model tends to converge at 99.24. Still unable to reach 99.4%

### Model8 With Data Augmentation
###### Target
Improve accuracy - Same as model 8, add data augmentation
###### Result
Model Parameters - 4,838
Train Accuracy - 99.13%
Test accuracy - 99.32%%
##### Analysis
Way better in terms of validation accracy. However, model tends to converge at 99.24. Still unable to reach 99.4%

### Model9:
###### Target
Improve accuracy to meet target. Use Squeeze-Expand model. LROnPlateau + Data Augmentation
###### Result
Model Parameters - 7,356
Train Accuracy - 99.31%
Test accuracy - 99.46%%
##### Analysis
Reached our required target!
We go with this model



