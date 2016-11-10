import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._

object Test extends App {
import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import breeze.numerics._
import breeze.stats._

  val (initialTheta1, initialTheta2) = Data.getWeightData()
  val (trainingData, classifications) = Data.getTrainingData()

  //--------------------------------------------------------------------------------
  // Initial cost with the initial theta1,2 and lambda =0.
  //--------------------------------------------------------------------------------  
  val (cost, theta1Gradient, theta2Gradient) = CostFunction.nnCostFunction(
    initialTheta1,
    initialTheta2,
    Data.INPUT_LAYER_SIZE,
    Data.HIDDEN_LAYER_SIZE,
    Data.NUM_LABELS,
    trainingData,
    classifications,
    0 /* lambda */ )

  println("Cost (lambda 0) is %s (this value should be about 0.287629)".format(cost));
  //println("Theat2 gradient is \n%s".format(theta2Gradient(0 to 9, 0 to 2)))
  //println("Theat1 gradient is \n%s".format(theta1Gradient(0 to 24, 0 to 2)))

  //--------------------------------------------------------------------------------
  // Initial cost with the initial theta1,2 and lambda = 1.
  //--------------------------------------------------------------------------------  

  val (cost1, t1g1, t2g1) = CostFunction.nnCostFunction(
    initialTheta1,
    initialTheta2,
    Data.INPUT_LAYER_SIZE,
    Data.HIDDEN_LAYER_SIZE,
    Data.NUM_LABELS,
    trainingData,
    classifications,
    1 /* lambda */ )

  println("Cost (lambda 1) is %s (this value should be about 0.383770)\n".format(cost1));
  //  println("Theat2 gradient is \n%s".format(t2g1(0 to 9, 0 to 2)))
  //  println("Theat1 gradient is \n%s".format(t1g1(0 to 24, 0 to 2)))

  //--------------------------------------------------------------------------------
  // Initial cost with the initial theta1,2 and lambda = 3.
  //--------------------------------------------------------------------------------  
  val (cost3, t1g3, t2g3) = CostFunction.nnCostFunction(
    initialTheta1,
    initialTheta2,
    Data.INPUT_LAYER_SIZE,
    Data.HIDDEN_LAYER_SIZE,
    Data.NUM_LABELS,
    trainingData,
    classifications,
    3 /* lambda */ )

  println("Cost (lambda 1) is %s (this value should be about 0.576051)\n".format(cost3));
  //  println("Theat2 gradient is \n%s".format(t2g3(0 to 9, 0 to 2)))
  //  println("Theat1 gradient is \n%s".format(t1g3(0 to 24, 0 to 2)))

}