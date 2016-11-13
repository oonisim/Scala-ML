import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._

object Predict extends App {
  val (theta1, theta2) = Data.getWeightData()
  val Z = Data.getValidationData()
  DisplayData(Z)
  
  //----------------------------------------------------------------------
  // Add bias column.
  //----------------------------------------------------------------------
  val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](Z.rows, 1),Z)
 
  //------------------------------------------------------------------------
  // Calculate the logistic value of the hidden layer H / Activation 2 (A2). 
  //------------------------------------------------------------------------
  val H_NET = X * theta1.t;
  val H_SIG = sigmoid(H_NET); // Activation
  val H_OUT = DenseMatrix.horzcat(DenseMatrix.ones[Double](H_SIG.rows, 1), H_SIG) // Add bias

  //------------------------------------------------------------------------
  // Calculate the logistic value of the output layer O / Activation 3 (A3).
  //------------------------------------------------------------------------
  val O_NET = H_OUT * theta2.t;
  val O_OUT = sigmoid(O_NET);
  println(O_OUT.toString(1000, 1000))
}