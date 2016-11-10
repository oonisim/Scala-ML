import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._

object CostFunction extends App {
  /**
   * ================================================================================
   * Provide a Evaluation/Cost function for Optimizer to search for the optimal input(s).
   * [Return]
   * Function 
   * DenseVector(Theta1 + Theta2) => (Cost, Gradient/Theta1 + Gradient/Theta2)
   * ================================================================================
   */
  def getInstance(
    input_layer_size: Int,
    hidden_layer_size: Int,
    number_of_labels: Int,
    training_data: DenseMatrix[Double],
    classifications: DenseMatrix[Int],
    lambda: Double): (DenseVector[Double]) => (Double, DenseVector[Double]) = {

    def f(theta12: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val (theta1, theta2) = Data.reshapeTheta12(theta12)
      val (cost, theta1grad, theta2grad) = nnCostFunction(
        theta1,
        theta2,
        input_layer_size,
        hidden_layer_size,
        number_of_labels,
        training_data,
        classifications, 
        lambda)
      
      (cost, Data.serializeTheta12(theta1grad, theta2grad))
    }
    f
  }
  /**
   * ================================================================================
   * Neural network cost and gradient function.
   * [Mechanism]
   * For the input set X, the nnCostFunction functions as:
   * [Theta1, Theta2] => [Cost(Theta1, Theta2), Delta(Cost/Theta1), Delta(Cost/Theta2)]
   *
   * [Return]
   * Tuple (Cost, Cost Gradient/Theat1, Cost Gradient/Theat2)
   * ================================================================================
   */
  def nnCostFunction(
    theta1: DenseMatrix[Double],
    theta2: DenseMatrix[Double],
    input_layer_size: Int,
    hidden_layer_size: Int,
    number_of_labels: Int,
    training_data: DenseMatrix[Double],
    classifications: DenseMatrix[Int],
    lambda: Double): (Double, DenseMatrix[Double], DenseMatrix[Double]) = {

    // Number of trainig data
    val m = training_data.rows

    //----------------------------------------------------------------------
    // Add bias column.
    //----------------------------------------------------------------------
    val X = DenseMatrix.horzcat(
      DenseMatrix.ones[Double](training_data.rows, 1),
      training_data)
    //println("%d, %d".format(X.rows, X.cols))

    //------------------------------------------------------------------------
    // Calculate the logistic value of the hidden layer H / Activation 2 (A2). 
    //------------------------------------------------------------------------
    val H_NET = X * theta1.t;
    val H_SIG = Utility.sigmoid(H_NET); // Activation
    val H_OUT = DenseMatrix.horzcat(DenseMatrix.ones[Double](H_SIG.rows, 1), H_SIG) // Add bias

    //------------------------------------------------------------------------
    // Calculate the logistic value of the output layer O / Activation 3 (A3).
    //------------------------------------------------------------------------
    val O_NET = H_OUT * theta2.t;
    val O_OUT = sigmoid(O_NET);

    //------------------------------------------------------------------------
    // Convert y (10, 10, 10 .... 9, 9, 9, ... ,1) into a boolean matrix.
    // if y(i) is 10, then E(i, :) is [1,0,0,0,0,0,0,0,0,0,0].
    //------------------------------------------------------------------------
    val Y = Data.getClassifictionMatrix(classifications)
    val J = cost(theta1, theta2, input_layer_size, hidden_layer_size, number_of_labels, training_data, classifications, lambda, X, Y, H_OUT, O_OUT)
    val (theta1_gradient, theta2_gradient) = gradient(theta1, theta2, input_layer_size, hidden_layer_size, number_of_labels, training_data, classifications, lambda, X, Y, H_OUT, O_OUT)

    (J, theta1_gradient, theta2_gradient)
  }
  
  /**
   * ================================================================================
   * Calculate cost for the theta given.
   * ================================================================================
   */
  def cost(
    theta1: DenseMatrix[Double],
    theta2: DenseMatrix[Double],
    input_layer_size: Int,
    hidden_layer_size: Int,
    number_of_labels: Int,
    training_data: DenseMatrix[Double],
    classifications: DenseMatrix[Int],
    lambda: Double,
    X: DenseMatrix[Double],
    Y: DenseMatrix[Int],
    H_OUT: DenseMatrix[Double],
    O_OUT: DenseMatrix[Double]): (Double) = {

    //------------------------------------------------------------------------
    // Calculate the cost at output without regularization.
    //------------------------------------------------------------------------
    val m = training_data.rows
    val one = DenseMatrix.fill(m, number_of_labels)(1.0)

    // Each row of (Y .* log(O)) is the cost at each output node for input xi. 
    // Take the sum of all columns in a row by sum(v, Axis._1 for the cost of each xi.
    val cost_y1 = -1.0 * sum(Y.map(_.toDouble) :* log(O_OUT), Axis._1) / m.toDouble
    val cost_y0 = -1.0 * sum((one - Y.map(_.toDouble)) :* log(one - O_OUT), Axis._1) / m.toDouble

    println("cost_y1 is %s".format(cost_y1(0 to 9)))

    // Each row of (cost_y1 + cost_y0) is the cost of xi. 
    // Take the sum of all rows for the total cost (xi: i = 1,2,3..)
    val J_OUT = sum(cost_y1 + cost_y0);
    println("Cost without regularization is %s".format(J_OUT))
    //------------------------------------------------------------------------
    // Regularize the cost.
    // Note that you should not be regularizing the terms that correspond to the bias. 
    // For the matrices Theta1 and Theta2, this corresponds to the first column.
    //------------------------------------------------------------------------
    val Theta2_square = theta2(::, 1 to -1) :^ 2.0
    val Theta2_reg = sum(Theta2_square) * lambda / (2 * m)
    val Theta1_square = theta1(::, 1 to -1) :^ 2.0
    val Theta1_reg = sum(Theta1_square) * lambda / (2 * m)
    val J = J_OUT + (Theta2_reg + Theta1_reg)

    println("The cost is %s".format(J))
    J
  }
  /**
   * ================================================================================
   *  MATLAB bfxfun similar function.
   *  Pick each element in OUT array, and multiply IN array with it.
   *  [In, Out]
   *  IN1   OUT1
   *  IN2   OUT2
   *  IN3   OUT3
   *
   *  [Return]
   *  IN1 x OUT1, IN2 x OUT1, IN3 x OUT1
   *  IN1 x OUT2, IN2 x OUT2, IN3 x OUT2
   *  IN1 x OUT3, IN2 x OUT3, IN3 x OUT3
   * ================================================================================
   */
  def bsxfun(in: DenseVector[Double], out: DenseVector[Double]): DenseMatrix[Double] = {
    val vectors = for {
      o <- out.toArray
    } yield {
      in.toArray.map { _ * o }
    }
    val matrix = DenseMatrix(vectors: _*)
    matrix
  }

  /**
   * ================================================================================
   * Compute the cost gradients Theta1_grad and Theta2_grad.
   * Return the partial derivatives of the cost function with respect to Theta1 and Theta2.
   * Check that your implementation is correct by running checkNNGradients.
   * ================================================================================
   */
  def gradient(
    theta1: DenseMatrix[Double],
    theta2: DenseMatrix[Double],
    input_layer_size: Int,
    hidden_layer_size: Int,
    number_of_labels: Int,
    training_data: DenseMatrix[Double],
    classifications: DenseMatrix[Int],
    lambda: Double,
    X: DenseMatrix[Double],
    Y: DenseMatrix[Int],
    H_OUT: DenseMatrix[Double],
    O_OUT: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val m = training_data.rows

    //------------------------------------------------------------------------
    // Calculate the gradients of Theta2
    //------------------------------------------------------------------------    
    val Theta2_grad = {
      def f(i: Int): DenseMatrix[Double] = {
        val yi = Y(i, ::).t.map(_.toDouble);
        // hi is the i th output of the hidden layer. H(i, :) is 26 data.
        val hi = H_OUT(i, ::).t;
        // oi is the i th output layer. O(i, :) is 10 data.
        val oi = O_OUT(i, ::).t;
        bsxfun(hi, (oi - yi))
      }
      val zero = DenseMatrix.zeros[Double](theta2.rows, theta2.cols);
      (0 until m).foldLeft(zero)((tg, i) => tg + f(i))
    }

    //------------------------------------------------------------------------
    // Calculate the gradients of Theta1
    //------------------------------------------------------------------------
    // Derivative of g(z): g'(z)=g(z)(1-g(z)) where g(z) is sigmoid(H_NET).
    // Theta1_grad = Theta1_grad + bsxfun(xi, delta_theta1));
    var Theta1_grad = DenseMatrix.zeros[Double](theta1.rows, theta1.cols);
    for (i <- (0 until m)) {
      // i is training set index of X (including bias). X(i, :) is 401 data.
      // xi is the input.
      val xi = X(i, ::).t;
      // yi is the classification. 
      val yi = Y(i, ::).t.map(_.toDouble);
      // hi is the i th output of the hidden layer. H(i, :) is 26 data.
      val hi = H_OUT(i, ::).t;
      // oi is the i th output layer. O(i, :) is 10 data.
      val oi = O_OUT(i, ::).t;

      //------------------------------------------------------------------------
      // Iterative version to calculate the gradients of Theta1
      //------------------------------------------------------------------------
      // Input layer index alpha (including bias) for Theat1_grad(j, alpha)
      // Hidden layer index j (including bias). There is no input into H0, hence there is no theta for H0. Remove H0.
      // Output layer index k 
      //------------------------------------------------------------------------
      def hf(j: Int) = {
        (hi(j) * (1 - hi(j)))
      }
      def ef(j: Int): Double = {
        (0 until oi.length) // k
          .foldLeft(0.0)((e, k) => e + (theta2(k, j) * (oi(k) - yi(k))))
      }
      for (j <- (1 until hi.length)) {
        for (alpha <- (0 until xi.length)) {
          val grad = xi(alpha) * hf(j) * ef(j)
          Theta1_grad(j - 1, alpha) = Theta1_grad(j - 1, alpha) + grad
        }
      }
    }

    // Regularization with the cost function and gradients.
    val Theta2_grad_reg = DenseMatrix.horzcat(DenseMatrix.zeros[Double](theta2.rows, 1), theta2(::, 1 to -1)) * (lambda / m)
    val Theta1_grad_reg = DenseMatrix.horzcat(DenseMatrix.zeros[Double](theta1.rows, 1), theta1(::, 1 to -1)) * (lambda / m)
    (((Theta1_grad / m.toDouble) + Theta1_grad_reg), ((Theta2_grad / m.toDouble) + Theta2_grad_reg))
  }
}
