package pl.wroc.uni.ii.evolution.objectivefunctions.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Sphere objective function for real vector individuals.
 * <p>
 * Calculates sum of squared differences between provided optimum value. The
 * calculated result is multiplied by -1 to conform Wevo convention on objective
 * functions.
 * </p>
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvSphereFunction implements
    EvObjectiveFunction<EvRealVectorIndividual> {

  /**
   * {@inheritDoc}
   */
  private static final long serialVersionUID = 1L;

  /**
   * Optimal individual values.
   */
  private final double[] optimum;


  /**
   * Creates new sphere function with optimum in <code>dimension_</code>-dimensional
   * zero point.
   * 
   * @param dimension_ accepted dimension of individuals
   */
  public EvSphereFunction(final int dimension_) {
    // new double[] contains 0.0 on all positions
    this(new double[dimension_]);
  }


  /**
   * Crates new sphere objective function with given optimum.
   * 
   * @param optimum_ Optimal individual values
   */
  public EvSphereFunction(final double[] optimum_) {
    this.optimum = optimum_;
  }


  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvRealVectorIndividual individual) {
    if (individual.getDimension() != optimum.length) {
      throw new IllegalArgumentException("Invalid dimension of individual");
    }

    double value = 0.0;
    for (int i = 0; i < individual.getDimension(); i++) {
      value -= Math.pow(individual.getValue(i) - optimum[i], 2.0);
    }

    return value;
  }
}
