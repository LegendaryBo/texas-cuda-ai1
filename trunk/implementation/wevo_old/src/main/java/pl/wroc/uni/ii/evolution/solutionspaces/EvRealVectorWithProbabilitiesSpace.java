package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvRealVectorWithProbabilitiesIndividuals
 * 
 * @author Tomasz Kozakiewicz, Lukasz Witko
 */
public class EvRealVectorWithProbabilitiesSpace implements
    EvSolutionSpace<EvRealVectorWithProbabilitiesIndividual> {

  private static final long serialVersionUID = 4555339322017778867L;

  protected int dimension;

  private EvObjectiveFunction<EvRealVectorWithProbabilitiesIndividual> objective_function;


  /**
   * @param i number of individuals chromosome dimension (length).
   */
  public EvRealVectorWithProbabilitiesSpace(
      EvObjectiveFunction<EvRealVectorWithProbabilitiesIndividual> objective_function,
      int i) {
    dimension = i;
    setObjectiveFuntion(objective_function);
  }


  /**
   * @return true if only individual is from correct class and dimension
   */
  public boolean belongsTo(EvRealVectorWithProbabilitiesIndividual individual) {
    return true;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvRealVectorWithProbabilitiesIndividual>> divide(
      int n) {
    return null;
  }


  /**
   * Generates random RealVectorWithProbabilitiesIndividuals.
   */
  public EvRealVectorWithProbabilitiesIndividual generateIndividual() {
    EvRealVectorWithProbabilitiesIndividual individual =
        new EvRealVectorWithProbabilitiesIndividual(dimension);
    for (int i = 0; i < individual.getDimension(); i++) {
      individual.setProbability(i, EvRandomizer.INSTANCE.nextGaussian());
      individual.setValue(i, Math.random());
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
  public EvRealVectorWithProbabilitiesIndividual takeBackTo(
      EvRealVectorWithProbabilitiesIndividual individual) {
    return individual;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvRealVectorWithProbabilitiesIndividual>> divide(
      int n, Set<EvRealVectorWithProbabilitiesIndividual> p) {
    return null;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvRealVectorWithProbabilitiesIndividual> objective_function) {
    if (objective_function == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    }
    this.objective_function = objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvRealVectorWithProbabilitiesIndividual> getObjectiveFuntion() {
    return objective_function;
  }
}
