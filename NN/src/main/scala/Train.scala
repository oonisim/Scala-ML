import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._
import breeze.optimize.FirstOrderMinimizer.OptParams

/**
 * Train the neural network for an near optimal weights.
 */
object Train extends App {
  val MAX_ITERATIONS = 100
  
  //val (initialTheta1, initialTheta2) = Data.getWeightData()
  val initialTheta1 = Data.randInitializeWeights(Data.INPUT_LAYER_SIZE, Data.HIDDEN_LAYER_SIZE)
  val initialTheta2 = Data.randInitializeWeights(Data.HIDDEN_LAYER_SIZE, Data.OUTPUT_LAYER_SIZE)
  val (trainingData, classifications) = Data.getTrainingData()
  val lambda = 0

  // val costFunction = CostFunctionImmutable.getInstance(       
  val costFunction = CostFunction.getInstance( 
    Data.INPUT_LAYER_SIZE,
    Data.HIDDEN_LAYER_SIZE,
    Data.NUM_LABELS,
    trainingData,
    classifications,
    lambda)

  val optimizer = Optimizer.getInstance(costFunction)
  //println(optimizer.getClass)
  
  //------------------------------------------------------------------------
  // Find a near optimal theta that minimize the cost function.
  //------------------------------------------------------------------------
  val initial = Data.serializeTheta12(initialTheta1, initialTheta2)
  val min = minimize(
      optimizer, 
      initial, 
      new OptParams(maxIterations=MAX_ITERATIONS))
  
  val (t1, t2) = Data.reshapeTheta12(min)
  print(t2)
  Data.putWeightData(t1, t2)
 
}
