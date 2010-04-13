package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Random;
import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvNaturalNumberVectorIndividuals
 * 
 * @author Marcin Golebiowski
 */
public class EvNaturalNumberVectorSpace implements
    EvSolutionSpace<EvNaturalNumberVectorIndividual> {

  private static final long serialVersionUID = -1312379527658064511L;

  private int dimension;

  private int max_value = Integer.MAX_VALUE;

  private EvObjectiveFunction<EvNaturalNumberVectorIndividual> objective_function;


  /**
   * @param i number of individuals chromosome dimension (length).
   */
  public EvNaturalNumberVectorSpace(
      EvObjectiveFunction<EvNaturalNumberVectorIndividual> objective_function,
      int i) {
    this(objective_function, i, Integer.MAX_VALUE);
  }


  public EvNaturalNumberVectorSpace(
      EvObjectiveFunction<EvNaturalNumberVectorIndividual> fun, int length,
      int max_value) {
    this.dimension = length;
    this.max_value = max_value;
    this.objective_function = fun;
  }


  /**
   * @return true if only individual is from correct class and dimension
   */
  public boolean belongsTo(EvNaturalNumberVectorIndividual individual) {
    return individual.getDimension() == dimension;
  }


  public Set<EvSolutionSpace<EvNaturalNumberVectorIndividual>> divide(int n) {

    return null;
  }


  public Set<EvSolutionSpace<EvNaturalNumberVectorIndividual>> divide(int n,
      Set<EvNaturalNumberVectorIndividual> p) {
    return null;
  }


  /**
   * Generates random NaturalNumberIndividual.
   */
  public EvNaturalNumberVectorIndividual generateIndividual() {
    return this.generateIndividual(max_value);
  }


  public EvNaturalNumberVectorIndividual generateIndividual(int max) {
    EvNaturalNumberVectorIndividual individual =
        new EvNaturalNumberVectorIndividual(dimension);
    for (int i = 0; i < individual.getDimension(); i++) {
      individual.setNumberAtPosition(i, EvRandomizer.INSTANCE.nextInt(max));
    }
    individual.setObjectiveFunction(objective_function);
    return individual;
  }


  /** Gets dimension (length) of individuals chromosome */
  public int getDimension() {
    return dimension;
  }


  /**
   * If only individual is from correct class and dimension, return this
   * individual itself. In other case throw exception.
   */
  public EvNaturalNumberVectorIndividual takeBackTo(
      EvNaturalNumberVectorIndividual individual) {
    return individual;
  }


  /**
   * @return uniform random number in range[0,1]
   */
  public static int uniformRandom() {
    Random rand = new Random();
    return rand.nextInt();
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvNaturalNumberVectorIndividual> objective_function) {
    this.objective_function = objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvNaturalNumberVectorIndividual> getObjectiveFuntion() {
    return objective_function;
  }
}
