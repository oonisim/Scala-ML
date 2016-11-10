import scala.math._
import breeze.linalg._
import breeze.stats._
import breeze.optimize._

object Utility extends{
  def sigmoid(z: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.tabulate[Double](z.rows, z.cols)(
      (r, c) => (1.0 / (1 + scala.math.exp(-1 * z(r, c)))))
  }

  def createBitMatrix(rows: Int, cols: Int, i: Int, j: Int) : DenseMatrix[Double] = {
    // Set 1.0 on (row, col) = (i, j) cell.
    val vector = for (r <- (0 until rows)) yield {
      if(r == i){
        Array.fill(cols){0.0}.toList.updated(j, 1.0).toArray
      }
      else Array.fill(cols){0.0}
    }
    val matrix = DenseMatrix(vector : _*)
    matrix
  }
}