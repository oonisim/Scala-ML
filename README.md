# Machine learning in Scala

## NN
[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning/home) Neural Network [Back Propagation](https://www.coursera.org/learn/machine-learning/home/week/5) implementation in Scala with [Scala NLP Breeze](https://github.com/scalanlp/breeze). Optimal value search fmincg is implemented with [Breeze.optimize](https://github.com/scalanlp/breeze/wiki/Quickstart#breezeoptimize) in Optimizer.scala.

Some MATLAB conventions may remain as it is used at Coursera, such as index is from 1, NOT 0. For the labels to classify digit, digit 1 is mapped to index 1, digit 0 is mapped to index 10. In Scala implementation, digit 1 is mapped to index 0, and digit 0 is mapped to index 9.

### Back Propagation from output layer to Theta2 

![Backpopagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta2Gradient.png)

### Back Propagation from hidden layer to Theta1 

![Backpropagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta1Gradient.png)

### Derivative Calculus

![Derivative](https://github.com/oonisim/Scala-ML/blob/master/NN/BPGradientCalculation.png)

### Training

The original data contains 5000 digit images. Used 4900 data and 100 iterations of gradients for training the network, and used the remaining 100 for validations.

### Prediction

Fed the validation data [Z.CSV](https://github.com/oonisim/Scala-ML/blob/master/NN/src/main/resources/Z.csv) to test as in the [Prediction results](https://youtu.be/3Oex8lODuLY) at YouTube.

