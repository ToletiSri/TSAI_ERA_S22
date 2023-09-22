# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 18

This folder consists of Assignment-18 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-18
With this assignment, we move towards generative AI training.  The assignment consists of 2 parts:

- Part 1 (UNETs):
    --  Train your own UNet from scratch, you can use the dataset and strategy provided in this [link](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406/) . You could also use the [dice loss function/UNET code](https://canvas.instructure.com/courses/6743641/assignments/37210228?module_item_id=94020701) discussed in the class. However, you need to train it 4 times with the following strategies
    1. MaxPooling + Transpose Convolution + Binary Cross Entropy loss
    2. MaxPooling + Transpose Convolution + Dice Loss
    3. Strided Convolution + Transpose Convolution + Binary Cross Entropy loss
    4.  Strided Convolution + Upsampling + Dice Loss
    
- Part 2 (VAEs)
-- Use this [VAE code](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing) for reference. However, you need to train the input image along with the label. You could take a combination of input/label as the input to the encoder block or both encoder and decoder. For inferencing, use an image with wrong label and see if your output starts looking like the wrong label. Repeat the iteration for 25 times to get 25 outputs. 
HINT: You need to have some percent of incorrect input-label pairs duting your training. 
-- Repeat this for 2 datasets - MNIST and CIFAR-10



### RESULTS:

- Part 1:
 TBD

- Part 2: 
-- MNIST: 
An image of digit-5, generated using VAE:

[![](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/Digit5.png)](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/Digit5.png)

An image of digit-5, being trained with wrong label - 9, over 25 iterations:

[![](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/Digit5ToDigit9.png)](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/Digit5ToDigit9.png)




-- CIFAR10:

An image of horse, generated using VAE:

[![](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/horse.png)](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/horse.png)

An image of horse being trained with wrong label - 'Bird', over 25 iterations
(You can see that the legs of the horse start to disappear by 25th iteration)

[![](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/HorseToBird.png)](https://raw.githubusercontent.com/ToletiSri/TSAI_ERA_Assignments/main/S18/Part2/ImagesForReadme/HorseToBird.png)

