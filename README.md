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
Feed the test data [Z.CSV](https://github.com/oonisim/Scala-ML/blob/master/NN/src/main/resources/Z.csv) of the validation image. The image is rotated 90 right and flipped holizontal to be MATLAB(?) format.

![Hand written image](https://raw.githubusercontent.com/oonisim/Scala-ML/master/NN/src/main/resources/9.bmp)

The actual displayed image (20 x 20 grid is added).

![Displayed Image:](https://raw.githubusercontent.com/oonisim/Scala-ML/master/NN/src/main/resources/Z.bmp)

**Result**

	6.140126526715231E-5    <----- 1
	2.853708343792005E-5    <----- 2
	7.375562958742116E-6
	7.57989103737568E-4
	4.098422401371338E-8
	0.0044276023643641275
	0.023448224138242442
	0.0017604837614926371
	0.8736362528027787      <----- 9 *
	3.0172795270422606E-4   <----- 0

