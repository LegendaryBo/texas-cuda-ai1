package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Objective function for knapsack problem. In this version we can't divide any
 * thing that we could insert into knapsack exactly his capacity weight.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvKnapsackProblem implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {
  /**
   * 
   */
  private static final long serialVersionUID = 1553146389829982158L;

  /**
   * Weight of each element. For Example: first element is 2kg So weight[0] = 2;
   */
  private final double[] weight;

  /**
   * Value of each element. For Example: first value is 23$ So value[0] = 23;
   */
  private final double[] value;

  /**
   * Capacity of knapsack. For example - we can put into 29kg So capacity - 29;
   */
  private final double capacity;


  /**
   * Create objective function.
   * 
   * @param weight_ Weight of each element
   * @param value_ Value of each element
   * @param capacity_ Capacity of knapsack
   */
  public EvKnapsackProblem(final double[] weight_, final double[] value_,
      final double capacity_) {
    if ((weight_ == null) || (value_ == null) || (capacity_ <= 0)
        || (weight_.length != value_.length)) {
      throw new IllegalArgumentException(
          "Length of weight and value must be equal "
              + "and capacity must be bigger than 0");
    }
    this.weight = weight_;
    this.value = value_;
    this.capacity = capacity_;
  }


  /**
   * @param individual individual
   * @return fitness
   */
  public double evaluate(final EvBinaryVectorIndividual individual) {
    int[] genes = individual.getGenes();
    if (genes.length != weight.length) {
      throw new IllegalArgumentException(
          "Length of indifidual's chromosone isn't equal to"
              + "length of weight");
    }
    double valueOfKnapsack = 0;
    double weightOfKnapsack = 0;
    for (int i = 0; i < genes.length; i++) {
      if ((genes[i] == 1) && (weightOfKnapsack + weight[i] < capacity)) {
        valueOfKnapsack += value[i];
        weightOfKnapsack += weight[i];
      }
    }
    return valueOfKnapsack;
  }

}