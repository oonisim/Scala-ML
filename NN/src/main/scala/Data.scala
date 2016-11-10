import java.io._
import java.nio.file.{ Paths, Files }
import breeze.linalg._

//================================================================================
// Training and weight data handling class
//================================================================================
object Data {
  val DATA_DIR = "D:/Home/Resources/Coursera/MachineLearning/W5.BackPropagation/assignment/src/main/resources"
  val SEPARATOR_CSV = ','
  val SEPARATOR_PATH = '/'

  val INPUT_LAYER_SIZE = 400; // Input Images of Digits
  val HIDDEN_LAYER_SIZE = 25;
  val NUM_LABELS = 10; // Note that digit "0" is mapped to label 10 due to MATLAB index is from 1.
  val OUTPUT_LAYER_SIZE = NUM_LABELS;

  /**
   * --------------------------------------------------------------------------------
   *  Get the hand written image data for the training.
   * --------------------------------------------------------------------------------
   *  [MATLAB data detail]
   *  X : 20 x 20 gray scale image of number (0-9)
   *  Y : Classification of the image (1-10) where 10 is to classify 0.
   *      To make things compatible with Octave/MATLAB indexing, where there is no zero index,
   *      mapped the digit zero to the value ten. Therefore, a "0" digit is labeled as 10",
   *      while the digits 1 to 9 are labeled as 1 to 9 in their natural order.
   *
   *  Name         Size                Bytes  Class     Attributes
   *  X         5000x400            16000000  double
   *  y         5000x1                 40000  double
   * --------------------------------------------------------------------------------
   */
  val TRAINING_DATA_X_FILE = "X.csv"
  val TRAINING_DATA_Y_FILE = "Y.csv"
  def getTrainingData(): (DenseMatrix[Double], DenseMatrix[Int]) = {
    val X_FILE = DATA_DIR + SEPARATOR_PATH + TRAINING_DATA_X_FILE
    val Y_FILE = DATA_DIR + SEPARATOR_PATH + TRAINING_DATA_Y_FILE
    require(
      Files.exists(Paths.get(X_FILE)) && Files.exists(Paths.get(Y_FILE)),
      "%s and/or %s does not exist".format(X_FILE, Y_FILE))

    val X = csvread(new File(X_FILE), SEPARATOR_CSV)
    val Y = csvread(new File(Y_FILE), SEPARATOR_CSV).map(_.toInt)
    println("Type of training data X is %s, %d x %d".format(X.getClass, X.rows, X.cols))
    println("Type of training data Y is %s, %d x %d".format(Y.getClass, Y.rows, Y.cols))

    (X, Y)
  }

  /**
   * --------------------------------------------------------------------------------
   *  Convert classifications Y (10, 10, 10 .... 9, 9, 9, ... ,1) into a boolean matrix.
   *  If Y(i) is 2, then M(i, :) is [0,0,1,0,0,0,0,0,0,0,0].
   * --------------------------------------------------------------------------------
   */
  def getClassifictionMatrix(Y: DenseMatrix[Int]): DenseMatrix[Int] = {
    val E = DenseMatrix.eye[Int](Data.NUM_LABELS)
    val classifications = Y(::, 0).toArray
    val vectors = for (i <- (0 until classifications.size)) yield {
      // E(i, ::) is Transpose(DenseVector) representing a columnar data, not a row/array.
      // Hence, transpose it to be a row/array data.

      // The original classification data is adjusted for MATLAB where the first index 1.
      // Instead of mapping number 0 to index 1, it was mapped to 10.
      // Therefore, need to shift left.
      // If the result matrix row 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, it means the match number is 1, NOT 0.
      // If the result matrix row 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, it means the match number is 0, NOT 9.
      // This odd order is used to calculate the cost yi * oi. 
      E(classifications(i) - 1, ::).t
    }
    // DenseMatrix can NOT be created from sequence of DenseVector.
    // Hence mapping a sequence of DenseVecor to sequence of arrays as intermediary.
    val matrix = DenseMatrix(vectors.map(_.toArray): _*)
    matrix
  }

  /**
   * --------------------------------------------------------------------------------
   *  Get weight at each neuron (theta)
   * --------------------------------------------------------------------------------
   *  Theta1 is in-beween input and hidden layer. Theta2 is hidden and output.
   *
   *  [MATLAB data detail]
   *  Name         Size             Bytes  Class     Attributes
   *  Theta1      25x401            80200  double
   *  Theta2      10x26              2080  double
   * --------------------------------------------------------------------------------
   */
  val WEIGHT_DATA_T1_FILE = "T1.csv"
  val WEIGHT_DATA_T2_FILE = "T2.csv"
  def getWeightData(): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val T1_FILE = DATA_DIR + SEPARATOR_PATH + WEIGHT_DATA_T1_FILE
    val T2_FILE = DATA_DIR + SEPARATOR_PATH + WEIGHT_DATA_T2_FILE
    require(
      Files.exists(Paths.get(T1_FILE)) && Files.exists(Paths.get(T2_FILE)),
      "%s and/or %s does not exist".format(T1_FILE, T2_FILE))

    val T1 = csvread(new File(T1_FILE), SEPARATOR_CSV)
    val T2 = csvread(new File(T2_FILE), SEPARATOR_CSV)
    println("Type of weight data T1 is %s, %d x %d".format(T1.getClass, T1.rows, T1.cols))
    println("Type of weight data T2 is %s, %d x %d".format(T2.getClass, T2.rows, T2.cols))

    (T1, T2)
  }

  def ones(rows: Int, cols: Int): DenseMatrix[Int] = {
    val s = for (i <- (0 until rows)) yield Array.fill(cols) { 1 }
    DenseMatrix(s: _*)
  }

  /**
   * --------------------------------------------------------------------------------
   *  Get weight by radom initialization.
   * --------------------------------------------------------------------------------
   */
  def randInitializeWeights(inputLayerSize: Int, outputLayerSize: Int): DenseMatrix[Double] = {
    val epsilon_init = 0.12;
    val W = DenseMatrix.rand(outputLayerSize, 1 + inputLayerSize)
    W :* 2 * epsilon_init - epsilon_init
  }

  /**
   * --------------------------------------------------------------------------------
   *  Flatten the theta1 and theta2 into a vector.
   * --------------------------------------------------------------------------------
   */
  def serializeTheta12(theta1: DenseMatrix[Double], theta2: DenseMatrix[Double]): DenseVector[Double] = {
    DenseVector.vertcat(theta1.toDenseVector, theta2.toDenseVector)
  }
  /**
   * --------------------------------------------------------------------------------
   *  Construct the theta1 and theta2 from a vector.
   * --------------------------------------------------------------------------------
   */
  def reshapeTheta12(in: DenseVector[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val theta1 = reshape(
      in(0 until HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)),
      HIDDEN_LAYER_SIZE,
      (INPUT_LAYER_SIZE + 1)); // 
    val theta2 = reshape(
      in((HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)) until in.length),
      OUTPUT_LAYER_SIZE,
      (HIDDEN_LAYER_SIZE + 1));
    (theta1, theta2)
  }
  def reshape(in: DenseVector[Double], rows: Int, cols: Int): DenseMatrix[Double] = {
    DenseMatrix.tabulate[Double](rows, cols)(
      (i, j) => in((j * (rows) + i)))
  }

}
