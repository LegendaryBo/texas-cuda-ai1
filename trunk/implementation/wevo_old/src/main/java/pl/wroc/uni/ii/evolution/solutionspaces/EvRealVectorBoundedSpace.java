package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Creates a space for {@link EvRealVectorIndividual}s with bounds on their
 * element values.
 * <p>
 * The bounds, both upper and lower are inclusive and stored as native
 * <code>double</code> values. If you wish not to restrict values on a certain
 * position in individuals, use {@link Double#MAX_VALUE} and
 * {@link Double#MIN_VALUE}, so that effectively there'll be no bound.
 * </p>
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvRealVectorBoundedSpace extends EvRealVectorSpace {

  /** {@inheritDoc} */
  private static final long serialVersionUID = 1L;

  /** Inclusive lower bound on individuals in this space. */
  private final double[] lower_bounds;

  /** Inclusive upper bound on individuals in this space. */
  private final double[] upper_bounds;


  /**
   * Creates new bounded continuous vector space with the same bounds on all
   * elements.
   * 
   * @param objective_function Objective function of generated individuals
   * @param dim Dimension of individuals in this space
   * @param lower_bound_ Lower bound on all elements in individuals, inclusive
   * @param upper_bound_ Upper bound on all elements in individuals, inclusive
   */
  public EvRealVectorBoundedSpace(
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function,
      final int dim, final double lower_bound_, final double upper_bound_) {
    super(objective_function, dim);
    if (lower_bound_ > upper_bound_) {
      throw new IllegalArgumentException(
          "Space's lower bound is greater than upper bound");
    }
    this.lower_bounds = new double[dim];
    Arrays.fill(this.lower_bounds, lower_bound_);
    this.upper_bounds = new double[dim];
    Arrays.fill(this.upper_bounds, upper_bound_);
  }


  /**
   * Creates new bounded continuous vector space with given bounds on elements.
   * <p>
   * The dimension of this space is determined from the length of bound arrays.
   * If array lengths don't match, an {@link IllegalArgumentException} is
   * thrown.
   * </p>
   * 
   * @param objective_function Objective function of generated individuals
   * @param lower_bounds_ Array of lower bounds (inclusive), where i-th value of
   *        array corresponds to lower bound of i-th value in individuals.
   * @param upper_bounds_ Array of upper bounds (inclusive), where i-th value of
   *        array corresponds to upper bound of i-th value in individuals.
   */
  public EvRealVectorBoundedSpace(
      final EvObjectiveFunction<EvRealVectorIndividual> objective_function,
      final double[] lower_bounds_, final double[] upper_bounds_) {
    super(objective_function, lower_bounds_.length);
    if (lower_bounds_.length != upper_bounds_.length) {
      throw new IllegalArgumentException(
          "Dimensions of lower and upper bound arrays do not match");
    }
    this.lower_bounds = lower_bounds_.clone();
    this.upper_bounds = upper_bounds_.clone();
  }


  /**
   * Checks if an individual belongs to this space.
   * 
   * @param individual individual to check
   * @return <code>true</code> if <code>individual</code> has correct
   *         dimension and its values between bounds of this space,
   *         <code>false</code> otherwise
   */
  @Override
  public boolean belongsTo(final EvRealVectorIndividual individual) {
    if (super.belongsTo(individual)) {
      for (int i = 0; i < this.getDimension(); i++) {
        double ind_value = individual.getValue(i);
        if (ind_value < lower_bounds[i] || ind_value > upper_bounds[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }


  /**
   * Generates new EvRealVectorIndividual with values selected randomly (with
   * uniform distribution) from within the bounds.
   * 
   * @return fresh individual with values (roughly) uniformly distributed in
   *         space bounds
   */
  @Override
  public EvRealVectorIndividual generateIndividual() {
    EvRealVectorIndividual individual =
        new EvRealVectorIndividual(this.getDimension());
    EvRandomizer randomizer = EvRandomizer.INSTANCE;
    for (int i = 0; i < this.getDimension(); i++) {
      // TODO (Krzysztof Sroka): replace with something possibly better, ie.
      // generating 0.0 and 1.0 values with the same probability as other.
      double next = randomizer.nextDouble();
      if (next == 0.0 && randomizer.nextBoolean()) {
        next = 1.0;
      }
      next = (next * (upper_bounds[i] - lower_bounds[i])) + lower_bounds[i];
      individual.setValue(i, next);
    }
    return individual;
  }


  /**
   * Returns EvRandomIndividual with values in space bounds.
   * 
   * @param individual individual to bring back to space
   * @return individual given as parameter, with values replaced (if needed) by
   *         upper/lower bound of this solution space
   */
  @Override
  public EvRealVectorIndividual takeBackTo(
      final EvRealVectorIndividual individual) {
    if (individual.getDimension() != this.getDimension()) {
      throw new IllegalArgumentException("Invalid dimension of individual");
    }

    for (int i = 0; i < this.getDimension(); i++) {
      double ind_value = individual.getValue(i);
      if (ind_value < lower_bounds[i]) {
        individual.setValue(i, lower_bounds[i]);
      } else if (ind_value > upper_bounds[i]) {
        individual.setValue(i, upper_bounds[i]);
      }
    }

    return individual;
  }
}
