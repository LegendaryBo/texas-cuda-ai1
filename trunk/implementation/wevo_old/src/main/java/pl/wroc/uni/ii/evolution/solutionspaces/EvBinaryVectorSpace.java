package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvBinaryVectorIndividual
 * 
 * @author Marek Chrusciel
 * @author Michal Humenczuk
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvBinaryVectorSpace implements
    EvSolutionSpace<EvBinaryVectorIndividual> {

  private static final long serialVersionUID = 5496084423283990032L;

  protected int dimension;

  protected EvObjectiveFunction<EvBinaryVectorIndividual> objective_function;


  /**
   * Constructs binary string with dimension given in argument. Objective
   * function is set with <code>goal_function</code> argument
   * 
   * @param <code>goal_function</code> objective function for individual
   * @param <code>dimension</code> number of individuals chromosome dimension
   *        (length).
   */
  public EvBinaryVectorSpace(
      EvObjectiveFunction<EvBinaryVectorIndividual> goal_function, int dimension) {

    this.dimension = dimension;

    objective_function = goal_function;
  }


  /**
   * Check if individual given with argument <code>individual</code> belongs
   * to this solution space. True if individual is binary and has the same
   * dimension.
   * 
   * @return if individual belongs to this solution space
   */
  public boolean belongsTo(EvBinaryVectorIndividual individual) {
    if (individual == null) {
      return false;
    }
    return individual.getDimension() == this.dimension;
  }


  /**
   * Gets dimension (length) of individuals chromosome
   * 
   * @return dimension of individual
   */
  public int getDimension() {
    return dimension;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvBinaryVectorIndividual> getObjectiveFuntion() {
    return objective_function;
  }


  /**
   * If only individual is from correct class and dimension, return this
   * individual itself. In other case return null.
   * 
   * @return binary individual itself or null
   */
  public EvBinaryVectorIndividual takeBackTo(EvBinaryVectorIndividual individual) {
    if (this.belongsTo(individual))
      return individual;
    else
      return null;
  }


  /**
   * Generates individual with fixed probability of each bit.
   * 
   * @return generated binary individual
   */
  public EvBinaryVectorIndividual generateIndividual(double[] probability_vector) {
    if (probability_vector.length != dimension) {
      throw new IllegalArgumentException(
          "Probability vector dimension must be equal to space dimension");
    }

    EvBinaryVectorIndividual bin_individual =
        new EvBinaryVectorIndividual(dimension);
    for (int i = 0; i < dimension; i++) {
      bin_individual.setGene(i, EvRandomizer.INSTANCE
          .nextProbableBooleanAsInt(probability_vector[i]));
    }

    bin_individual.setObjectiveFunction(getObjectiveFuntion());

    return bin_individual;
  }


  /**
   * Generate single individual of specified size and objective function.<BR>
   * Each gene is shuffled randomly.
   */
  public EvBinaryVectorIndividual generateIndividual() {

    EvBinaryVectorIndividual individual =
        new EvBinaryVectorIndividual(dimension);
    int individual_size = individual.getDimension();
    for (int i = 0; i < individual_size; i++) {
      individual.setGene(i, EvRandomizer.INSTANCE.nextInt(2)); // randomize 0
                                                                // or 1
    }
    individual.setObjectiveFunction(objective_function);
    return individual;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {
    if (objective_function == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    }
    this.objective_function = objective_function;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvBinaryVectorIndividual>> divide(int n) {
    return null;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvBinaryVectorIndividual>> divide(int n,
      Set<EvBinaryVectorIndividual> p) {
    return null;
  }

}
