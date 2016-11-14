import scala.math.sqrt
import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._
import breeze.plot._
import java.io._

object DisplayData extends App {
  // Take N images of (20 x 20) pixels from the training data.
  // Display N images in the frame of rows(Display frame row size) x cols(Display frame column size).
  def apply(X: DenseMatrix[Double]): DenseMatrix[Double] = {

    val numImages = X.rows.toInt
    val oneImage = X(0, ::).t

    val imageSize = oneImage.length
    val imageWidth = Math.sqrt(imageSize).toInt
    val imageHeight = imageSize / imageWidth

    val rows = sqrt(numImages).floor.toInt
    val cols = sqrt(numImages).floor.toInt

    var display = DenseMatrix.ones[Double](rows * imageHeight, cols * imageWidth)
    for (r <- 0 until rows; c <- 0 until cols) {
      val xi = (cols * r + c)
      val maxVal = max(X(xi, ::)).abs
      val image = (X(xi, ::).t)
      val cell = DenseMatrix.tabulate[Double](imageHeight, imageWidth) {
        (h, w) => image(h * imageHeight + w)
      }
      /*
      display(
        (r * imageHeight) until (r * imageHeight + imageHeight),
        (c * imageWidth) until (c * imageWidth + imageWidth)) := cell
*/
      for (w <- (0 until imageWidth); h <- (0 until imageHeight)) {
        //display((r * imageHeight) + h, (c * imageWidth) + w) = X(xi, h * imageHeight + w) / maxVal
        display(
          (r * imageHeight) + h, (c * imageWidth) + w) =
          X(xi, w * imageHeight + (imageHeight - h - 1))
      }
    }
    val f = Figure()
    f.subplot(0) += image(display)
    f.refresh()
    f.saveas("image.png")
    csvwrite(new File("image.csv"), display)

    display
  }

}