# Machine learning in Scala

## NN
[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning/home) Neural Network [Back Propagation](https://www.coursera.org/learn/machine-learning/home/week/5) implementation in Scala with [Scala NLP Breeze](https://github.com/scalanlp/breeze). Optimal value search fmincg is implemented with [Breeze.optimize](https://github.com/scalanlp/breeze/wiki/Quickstart#breezeoptimize) in Optimizer.scala.

### Back Propagation from output layer to Theta2 

![Backpopagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta2Gradient.png)

### Back Propagation from hidden layer to Theta1 

![Backpropagation Theta2](https://github.com/oonisim/Scala-ML/blob/master/NN/Theta1Gradient.png)

### Derivative Calculus

![Derivative](https://github.com/oonisim/Scala-ML/blob/master/NN/BPGradientCalculation.png)

### Prediction

After training (using 4900 data and 100 iterations), used the trained theta and fed the test data [Z.CSV](https://github.com/oonisim/Scala-ML/blob/master/NN/src/main/resources/Z.csv) of the validation images.

**Result**
MATLAB indexing starting from 1, NOT 0. Hence the original MATLAB mapping is digit 1 is mapped to index 1, digit 0 is mapped to index 10. Hence in Scala, digit 1 is mapped to index 0, and digit 0 is mapped to index 9.

[Prediction results](https://youtu.be/3Oex8lODuLY)
