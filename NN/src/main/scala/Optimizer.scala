import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._


/**
 * ================================================================================
 * Find an optimal input (theta1+theta2) for the evaluation/cost function of NN.
 * Using Breeze function optimization.
 * http://www.scalanlp.org/api/breeze/#breeze.optimize.DiffFunction
 * ================================================================================
 */
object Optimizer {
  // Get DiffFunction[f: DenseVector[Double] => (Double, DenseVector[Double]).
  // Stick to DenseVector[Double] for the cost function input/output for Breeze optimizer.
  def getInstance(cf: (DenseVector[Double]) => (Double, DenseVector[Double])): DiffFunction[DenseVector[Double]] = {
    val df = new DiffFunction[DenseVector[Double]] {
      def calculate(input: DenseVector[Double]) = cf(input)
    }
    df
  }
}
