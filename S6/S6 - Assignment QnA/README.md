# Learn Backpropogation by Implementing it From Scratch
This repository and its README serves as a tutorial of how to implement Backpropagation in Neural Networks from scratch. Please refer to the excel - [BackPropogation.xlsx](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/BackPropagation.xlsx)
## Neural Network

Consider the following network with initial set of weights assigned as shown
![](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/Network.jpg?raw=true)
Where,
- i1, i2 - Input kayers
- h1, h2 - hidden layers
- a_h1, a_h2 - Values after applying activation function on h1 and h2
- o1, o2 - Output layers
- a_o1, a_o2 - Values after applying activation function to o1 and o2,
- E1, E2 - Errors at the output layers
- E_total - total error. Refer to the formula in the image above
- [w] - weights
- Activation function used - [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)

## Forward Propgation Calculations

Let's consider the initial input values - i1 = 0.5 and i2 = 0.5. The expected output for this input set is  0.5(t1) and 0.5(t2). To arrive at the ideal weights for this network, for the expected input/out data(train data), let's initialise the network's weights with some random values. (Please refer to image above for these values.) 

We use the formulae below, to calculate - h1, h2, a_h1, a_h2, o1, o2, a_o1, a_o2, E1, E2, E_total
- h1 = (i1 * w1) + (i2 * w2)
- h2 = (i1 * w3) + (i2 * w4)
- a_h1 = sigmoid(h1)
- a_h2 = sigmoid(h2)
- o1 = (a_h1 * w5) + (a_h2 * w6)
- o2 = (a_h1 * w7) + (a_h2 * w8)
- a_o1 = sigmoid(o1)
- a_o2 = sigmoid(o2)
- E1 = 1/2*(t1 - a_o1)^2
- E2 = 1/2*(t2 - a_o2)^2
- E_total = E1+E2
(Refer to the [BackPropgation.xlsx](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/BackPropagation.xlsx) for the numerical values obtained)

## Backward Propogation Calculations
Once we have arrived at the error(E1 and E2) values, through the forward propogation calculations above, we need to readjust the weghts, such that the error values are minimised. The process of re-calculating the weights based on the errors obtained from the current weights is referred as - Backward propogation.

We calculate rate of change of error, w.r.t each weight, and multiply this with the learning rate to obtain the new weight
> w_new = w_old + (LearningRate) * (rate of change of error wrt w)

To calculate rate of change of error w.r.t w(Weights), we use partial differentiation and chain rule. Refer to formulae below:

![](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/NetworkAndFormulae.jpg?raw=true)

## Convergence

We now repeat the steps of forward and backward propogation calculations until the error is minised/converged towards 0.

Snippet from the BackwardPropogation table
![](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/BackPropogationCalculationTable.jpg?raw=true)
## Learning Rate
[Learning rate](https://en.wikipedia.org/wiki/Learning_rate) - refers to the step size at each iteration while moving towards a minimum loss (Error E).
Learning rate affects the number of iterations needed for converge. If the learning rate is too small, the convergence would take many iterations. However, if the learning rate is too high, the error could never converge as well.
Refer below to the plots below showing Loss v/s iteration number, for different learning rates. See how learning rate has an impact on the iterations needed for convergence.

Learning rate - 0.1:
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_0.1.jpg?raw=true) 


Learning rate - 0.2:
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_0.2.jpg?raw=true) 

Learning rate - 0.5:
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_0.5.jpg?raw=true) 

Learning rate - 1.0:
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_1.0.jpg?raw=true) 

Learning rate - 2.0:
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_2.0.jpg?raw=true) 

Learning rate - 1000(never converges):
![Learning Rate - 0.1](https://github.com/ToletiSri/TSAI_ERA_Assignments/blob/main/S6/S6%20-%20Assignment%20QnA/LearningRate_1000.jpg?raw=true) 

**That's all on this assignment. Hope you had a good learning time!**

