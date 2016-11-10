import scala.math._
import breeze.linalg._
import breeze.stats._
import breeze.optimize._

object Utility extends {
  def createBitMatrix(rows: Int, cols: Int, i: Int, j: Int): DenseMatrix[Double] = {
    // Set 1.0 on (row, col) = (i, j) cell.
    val vector = for (r <- (0 until rows)) yield {
      if (r == i) {
        Array.fill(cols) { 0.0 }.toList.updated(j, 1.0).toArray
      } else Array.fill(cols) { 0.0 }
    }
    val matrix = DenseMatrix(vector: _*)
    matrix
  }


}