import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._

object Train extends App {

  //val (initialTheta1, initialTheta2) = Data.getWeightData()
  val initialTheta1 = Data.randInitializeWeights(Data.INPUT_LAYER_SIZE, Data.HIDDEN_LAYER_SIZE)
  val initialTheta2 = Data.randInitializeWeights(Data.HIDDEN_LAYER_SIZE, Data.OUTPUT_LAYER_SIZE)
  val (trainingData, classifications) = Data.getTrainingData()
  val lambda = 0

  val costFunction = CostFunction.getInstance(
    Data.INPUT_LAYER_SIZE,
    Data.HIDDEN_LAYER_SIZE,
    Data.NUM_LABELS,
    trainingData,
    classifications,
    lambda)

  def f(xs: DenseVector[Double]) = sum(xs :^ 2.0)
  def gradf(xs: DenseVector[Double]) = 2.0 :* xs
  val xs = DenseVector.ones[Double](3)
  val optTrait = new DiffFunction[DenseVector[Double]] {
    def calculate(xs: DenseVector[Double]) = (f(xs), gradf(xs))
  }
  val minimum = minimize(optTrait, DenseVector(1.0, 1.0, 1.0))
  
  val optimizer = Optimizer.getInstance(costFunction)
  println(optimizer.getClass)
  
  val ini = Vector(initialTheta1, initialTheta2)
  val min = minimize(optimizer, ini)

}
