package pl.wroc.uni.ii.evolution.solutionspaces;


import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvRealVectorWithProbabilitiesIndividuals
 * 
 * @author Tomasz Kozakiewicz, Lukasz Witko
 */
public class EvMiLambdaRoKappaSpace 
          implements EvSolutionSpace<EvMiLambdaRoKappaIndividual> {;
          
  private static final long serialVersionUID = -5878716045284548338L;
  
  private EvObjectiveFunction<EvMiLambdaRoKappaIndividual> objective_function;  
  
  protected int dimension;
  
  /**
   * @param i number of individuals chromosome dimension (length).
   */
  public EvMiLambdaRoKappaSpace(EvObjectiveFunction<EvMiLambdaRoKappaIndividual> objective_function, int i) {
    setObjectiveFuntion(objective_function);
    dimension = i;
  }

  /**
   * @return true if only individual is from correct class and dimension
   */
  public boolean belongsTo(EvMiLambdaRoKappaIndividual individual) {
    return true;
  }
  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMiLambdaRoKappaIndividual>> divide(int n) {
    return null;
  }

  /**
   * Generates random RealVectorWithProbabilitiesIndividuals.
   */
  public EvMiLambdaRoKappaIndividual generateIndividual() {
    EvMiLambdaRoKappaIndividual individual = 
          new EvMiLambdaRoKappaIndividual(dimension);
    for(int i = 0; i < individual.getDimension() ; i++) {
      individual.setProbability( i, EvRandomizer.INSTANCE.nextGaussian() );
      individual.setValue(i, Math.random());
    }
    
    int alpha_size = individual.getAlphaLength();
    for(int i = 0; i < alpha_size ; i++) {
      individual.setAlpha( i, 2*Math.PI*EvRandomizer.INSTANCE.nextDouble() );
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
  public EvMiLambdaRoKappaIndividual takeBackTo(
      EvMiLambdaRoKappaIndividual individual) {
    return individual;
  }
  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMiLambdaRoKappaIndividual>> divide(
      int n, Set<EvMiLambdaRoKappaIndividual> p ) {
    return null;
  }

  /**
   * {@inheritDoc}
   */  
  public void setObjectiveFuntion(EvObjectiveFunction<EvMiLambdaRoKappaIndividual> objective_function) {
    if (objective_function == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    } 
    this.objective_function = objective_function;
  }

  public EvObjectiveFunction<EvMiLambdaRoKappaIndividual> getObjectiveFuntion() {
    return objective_function;
  }
}
