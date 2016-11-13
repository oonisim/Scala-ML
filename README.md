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
Usinng the trained theta after 100 interations, feed the test data [Z.CSV](https://github.com/oonisim/Scala-ML/blob/master/NN/src/main/resources/Z.csv) of the validation image. The image is rotated 90 right and flipped holizontal to be MATLAB(?) format.

![Hand written image](https://raw.githubusercontent.com/oonisim/Scala-ML/master/NN/src/main/resources/9.bmp)

The actual displayed image (20 x 20 grid is added).

![Displayed Image:](https://raw.githubusercontent.com/oonisim/Scala-ML/master/NN/src/main/resources/Z.bmp)

**Result**
MATLAB indexing starting from 1, NOT 0. Hence the original MATLAB mapping is digit 1 is mapped to index 1, digit 0 is mapped to index 10. Hence in Scala, digit 1 is mapped to index 0, and digit 0 is mapped to index 9.

	9.985166843331599E-4
	3.440359959890938E-4
	0.012995162855350469
	0.02014989450785369
	3.306913551558261E-6
	5.076212747509842E-6
	0.01631822486361171
	0.0017057245044301748
	0.71953085160519        <----- 9
	8.473379314208845E-7


