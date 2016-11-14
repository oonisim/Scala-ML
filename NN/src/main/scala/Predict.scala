import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._

/**
 * Predict numbers based on the trained neural network. 
 */
object Predict extends App {
  val (theta1, theta2) = Data.getWeightData()
  def _predict(x: DenseVector[Double]): Unit = {
    val image = x.toDenseMatrix
    DisplayData(image)
    Thread.sleep(1000)
    
    //----------------------------------------------------------------------
    // Add bias column.
    //----------------------------------------------------------------------
    val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](image.rows, 1), image)

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
    //println(O_OUT.toString(1000, 1000))
    
    
    val (activation, index) = argmax(O_OUT)
    val prediction = if(index == 9) 0 else (index +1)
    

    println("Prediction is %d, probability is %s".format(prediction, O_OUT.toArray(index)))
    Thread.sleep(3000)
  }
  

  val Z = Data.getValidationData()
  (0 until Z.rows).foreach(i => _predict(Z(i, ::).t))
}