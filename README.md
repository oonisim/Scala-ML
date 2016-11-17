# Machine learning in Scala

## NN (Neural network Handwriting Recognition)

[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning/home) Neural Network [Back Propagation](https://www.coursera.org/learn/machine-learning/home/week/5) implementation in Scala with [Scala NLP Breeze](https://github.com/scalanlp/breeze). Optimal value search fmincg is implemented with [Breeze.optimize](https://github.com/scalanlp/breeze/wiki/Quickstart#breezeoptimize) in Optimizer.scala.

![NN](https://github.com/oonisim/Scala-ML/blob/master/NN/TwoLayerNeuralNetBackpropagation.png)

Some MATLAB conventions may remain as it is used at Coursera, such as index is from 1, NOT 0. For the labels to classify digit, digit 1 is mapped to index 1, digit 0 is mapped to index 10. In Scala implementation, digit 1 is mapped to index 0, and digit 0 is mapped to index 9.

### Mechanism

Represent how well/poor the network(theta1, theta2) performs with the cost, which is basically how far away the output of the network is from the correct digit. Feed back the cost to adjust (theta1, theta2) to bring down the cost.

### Cost Function

Estimate the effectiveness of the theta, calculate the penalty/error, or cost.

![Cost Function](https://github.com/oonisim/Scala-ML/blob/master/NN/NNCostFunction.png)

### Back Propagation from output layer towards hidden layer

To feed the cost back and adjust the theta2 to bring the cost down, calculate the gradient of theta2. The optimization function from Breeze utilizes the gradient to run the gradient descent.

![Backpopagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta2Gradient.png)

### Back Propagation from hidden layer to Theta1 

To feed the cost back and adjust the theta1 to bring the cost down, calculate the gradient of theta1. The optimization function from Breeze utilizes the gradient to run the gradient descent.

![Backpropagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta1Gradient.png)

### Derivative Calculus

![Derivative](https://github.com/oonisim/Scala-ML/blob/master/NN/BPGradientCalculation.png)

### Training / Optimization

The original data contains 5000 digit images. Used 4900 data and 100 iterations of gradients for training the network, and used the remaining 100 for validations. Use the Breeze.optimize package DiffFuction to run the gradient descent(?).

### Prediction

Fed the validation data [Z.CSV](https://github.com/oonisim/Scala-ML/blob/master/NN/src/main/resources/Z.csv) to test as in the [Prediction results](https://youtu.be/3Oex8lODuLY) at YouTube.

![Result](https://github.com/oonisim/Scala-ML/blob/master/NN/Result.png)
