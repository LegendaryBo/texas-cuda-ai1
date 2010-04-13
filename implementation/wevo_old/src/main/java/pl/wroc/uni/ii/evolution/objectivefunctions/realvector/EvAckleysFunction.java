package pl.wroc.uni.ii.evolution.objectivefunctions.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Generalization of Ackley's function for multidimensional continuous spaces.
 * 
 * <p>
 * Ackley's function is a multimodal test function suitable for checking whether
 * an algorithm can avoid local optimum traps during the search. The range of
 * individual values should be restricted to an interval where differences in
 * function's value are significant.
 * </p>
 * 
 * <p>
 * Ackley's function for multidimensional is defined in Thomas Baecks'
 * <i>Evolutionary Algorithms in Theory and Practice</i> as:<br>
 * <code>f(x) = -c<sub>1</sub> * 
 * exp(-c<sub>2</sub> * 
 * sqrt((Sigma<sub>i=1..n</sub> x<sub>i</sub><sup>2</sup>)/n)) -
 * exp((Sigma<sub>i=1..n</sub> cos(c<sub>3</sub> * x<sub>i</sub>))/n) + 
 * c<sub>1</sub> + e</code><br>
 * This formula was based on D.H. Ackley's definition for two-dimensional
 * function. In wEvo we define the result of Ackley's function as
 * <code>-f(x)</code> to conform our convention. The global optimum of
 * Ackley's function is the vector
 * <code>x<sub>opt</sub> = &lt;0,...,0&gt;<sup>T</sup></code></p>
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvAckleysFunction implements
    EvObjectiveFunction<EvRealVectorIndividual> {

  /** Class serial version. */
  private static final long serialVersionUID = 6379310049713184700L;

  /** c<sub>1</sub> parameter of Ackley's function. */
  private final double c1;

  /** c<sub>2</sub> parameter of Ackley's function. */
  private final double c2;

  /** c<sub>3</sub> parameter of Ackley's function. */
  private final double c3;

  /**
   * Creates a new Ackley's function with given parameters.
   * 
   * <p>
   * Refer to this class description for definition of Ackley's function and its
   * parameters.
   * </p>
   * 
   * @param param_c1
   *          <code>c<sub>1</sub></code> of Ackley's function
   * @param param_c2
   *          <code>c<sub>2</sub></code> of Ackley's function
   * @param param_c3
   *          <code>c<sub>3</sub></code> of Ackley's function
   */
  public EvAckleysFunction(final double param_c1, final double param_c2,
      final double param_c3) {
    this.c1 = param_c1;
    this.c2 = param_c2;
    this.c3 = param_c3;
  }

  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvRealVectorIndividual individual) {
    // f(x) = c1 * e^exp_square + e^exp_cos - c1 - e 
    double exp_square = 0.0;
    double exp_cos = 0.0;

    for (int i = 0; i < individual.getDimension(); i++) {
      double ithval = individual.getValue(i);
      exp_square += ithval * ithval;
      exp_cos += Math.cos(c3 * ithval);
    }

    exp_square = -c2 * Math.sqrt(exp_square / individual.getDimension());
    exp_cos /= individual.getDimension();
    return c1 * Math.exp(exp_square) + Math.exp(exp_cos) - c1 - Math.E;
  }
}
