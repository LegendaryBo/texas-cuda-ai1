package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvRealVectorIndividuals.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvRealVectorSpace implements
    EvSolutionSpace<EvRealVectorIndividual> {

  /** {@inheritDoc} */
  private static final long serialVersionUID = 8279690263081096298L;

  /** Objective function for generated individuals. */
  private EvObjectiveFunction<EvRealVectorIndividual> objective_function;

  /** Dimension of this space. */
  private final int dimension;


  /**
   * Creates new EvRealVectorSpace with specified objective function and
   * dimension.
   * 
   * @param objective_function_ objective function for generated individuals.
   * @param i number of individuals chromosome dimension (length).
   */
  public EvRealVectorSpace(
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function_,
      final int i) {
    setObjectiveFuntion(objective_function_);
    dimension = i;
  }


  /**
   * Checks if individual belongs to this space.
   * 
   * @param individual individual to check
   * @return true if the given individual has correct dimension
   */
  public boolean belongsTo(final EvRealVectorIndividual individual) {
    return individual.getDimension() == this.dimension;
  }


  /**
   * Not implemented yet.
   * 
   * @see EvSolutionSpace#divide(int)
   * @param n unused
   * @return null
   */
  public Set<EvSolutionSpace<EvRealVectorIndividual>> divide(final int n) {
    // TODO (): implement
    return null;
  }


  /**
   * Not implemented yet.
   * 
   * @see EvSolutionSpace#divide(int, Set)
   * @param n unused
   * @param p unused
   * @return null
   */
  public Set<EvSolutionSpace<EvRealVectorIndividual>> divide(final int n,
      final Set<EvRealVectorIndividual> p) {
    // TODO (): implement
    return null;
  }


  /**
   * Generates random EvRealVectorIndividual.
   * 
   * @return fresh {@link EvRealVectorIndividual} with dimension equal to
   *         dimension of this space and values selected randomly (with uniform
   *         probability) from space of double values.
   */
  public EvRealVectorIndividual generateIndividual() {
    EvRealVectorIndividual individual = new EvRealVectorIndividual(dimension);
    for (int i = 0; i < individual.getDimension(); i++) {
      Double value = Double.MAX_VALUE * EvRandomizer.INSTANCE.nextDouble();
      if (EvRandomizer.INSTANCE.nextBoolean()) {
        value = -value;
      }
      individual.setValue(i, value);
    }
    individual.setObjectiveFunction(objective_function);
    return individual;
  }
  
  /**
   * Generates random EvRealVectorIndividual with normal distribution.
   * 
   * @param mean_vector expected values vector
   * @param deviation_vector standard deviations vector
   * @return fresh {@link EvRealVectorIndividual} with dimension equal to
   *         dimension of this space and values selected randomly (with normal
   *         distribution) from space of double values.
   */
  public EvRealVectorIndividual generateIndividual(final double[] mean_vector, 
                                             final double[] deviation_vector)  {
    if (mean_vector.length != dimension 
        || deviation_vector.length != dimension) {
      throw new IllegalArgumentException(
      "Mean and deviation vectors dimension must be equal to space dimension");
    }
    
    EvRealVectorIndividual individual = new EvRealVectorIndividual(dimension);
    for (int i = 0; i < individual.getDimension(); i++) {
      individual.setValue(i, 
        EvRandomizer.INSTANCE.nextGaussian(mean_vector[i], deviation_vector[i])
      );
    }
    
    individual.setObjectiveFunction(objective_function);
    
    return individual;

  }
  
  /**
   * Returns dimension of this space (and length of individual chromosomes).
   * 
   * @return dimension of this space
   */
  public int getDimension() {
    return dimension;
  }


  /**
   * If only individual has correct dimension, returns this individual itself.
   * In other case throws exception.
   * 
   * @param individual individual to bring back to space
   * @return given individual if it belongs to this space
   */
  public EvRealVectorIndividual takeBackTo(
      final EvRealVectorIndividual individual) {
    if (individual.getDimension() == this.dimension) {
      return individual;
    } else {
      throw new IllegalArgumentException("Invalid individual");
    }
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function_) {
    if (objective_function_ == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    }
    this.objective_function = objective_function_;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvRealVectorIndividual> getObjectiveFuntion() {
    return objective_function;
  }
}
