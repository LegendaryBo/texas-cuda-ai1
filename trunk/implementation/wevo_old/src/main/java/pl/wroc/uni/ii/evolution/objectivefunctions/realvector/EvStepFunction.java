package pl.wroc.uni.ii.evolution.objectivefunctions.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Step objective function for real vector individuals.
 * <p>
 * This function is a modification of the sphere function, by introducing
 * plateaus of objective function value, giving no gradient information for
 * directing the search. The optimal plateau contains all points having x<sub>i</sub>
 * in [opt<sub>i</sub>-0.5,opt<sub>i</sub>+0.5) for every i (where
 * <code>opt</code> is optimum value for created function).
 * </p>
 * <p>
 * Based on chapter 3.2 of <i>Evolutionary Algorithms in Theory and Practice
 * </i> by Thomas Back.
 * </p>
 * 
 * @see EvSphereFunction
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvStepFunction implements
    EvObjectiveFunction<EvRealVectorIndividual> {

  /** {@inheritDoc} */
  private static final long serialVersionUID = 1L;

  /** Optimal value of an individual. */
  private final double[] optimum;


  /**
   * Creates new step function with center of optimal plateau in
   * <code>dimension_</code>-dimensional zero point.
   * 
   * @param dimension_ dimension of accepted individuals.
   */
  public EvStepFunction(final int dimension_) {
    // new double[] contains 0.0 on all positions
    this(new double[dimension_]);
  }


  /**
   * Creates new step function with optimum at the provided value.
   * 
   * @param optimum_ center of optimal plateau
   */
  public EvStepFunction(final double[] optimum_) {
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
    // off MagicNumber
    // This magic number is added to each value to make optimal value
    // contained in this function occur in the middle of optimal plateau.
    double plateau_shift = 0.5;
    // on MagicNumber
    for (int i = 0; i < individual.getDimension(); i++) {
      double floored_val =
          Math.floor(individual.getValue(i) - optimum[i] + plateau_shift);
      value -= Math.pow(floored_val, 2.0);
    }

    return value;
  }
}
