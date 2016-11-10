import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._


/**
 * ================================================================================
 * Find an optimal input (theta1, theta2) for the evaluation/cost function of NN.
 * Using Breeze function optimization.
 * http://www.scalanlp.org/api/breeze/#breeze.optimize.DiffFunction
 * ================================================================================
 */
/*
object Optimizer {
  type THETA_SEQUENCE = (DenseMatrix[Double], DenseMatrix[Double]) 
  type COST_FUNCTION =  THETA_SEQUENCE => (Double, THETA_SEQUENCE)

  // Get DiffFunction[THETA_SEQUENCE] instance
  def getInstance(cf: COST_FUNCTION): DiffFunction[THETA_SEQUENCE] = {
    val df = new DiffFunction[THETA_SEQUENCE] {
      def calculate(input: THETA_SEQUENCE) = input match {
        case (theta1, theta2) => cf(theta1, theta2)
        case default => throw new Exception("Illeagal argument")
      }
    }
    df
  }
}
*/

object Optimizer {
  // Get DiffFunction[THETA_SEQUENCE] instance
  def getInstance(cf: (DenseMatrix[Double], DenseMatrix[Double])  => (Double, (DenseMatrix[Double], DenseMatrix[Double]))): DiffFunction[(DenseMatrix[Double], DenseMatrix[Double]) ] = {
    val df = new DiffFunction[(DenseMatrix[Double], DenseMatrix[Double]) ] {
      def calculate(input: (DenseMatrix[Double], DenseMatrix[Double]) ) = input match {
        case (theta1, theta2) => cf(theta1, theta2)
        case default => throw new Exception("Illeagal argument")
      }
    }
    df
  }
}