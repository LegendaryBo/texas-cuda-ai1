package pl.wroc.uni.ii.evolution.objectivefunctions.realvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

/**
 * Generalization of the Weierstrass function for multidimensional spaces.
 * 
 * <p>The definition of Weierstrass function used in Wevo is the one originally
 * by Weierstrass:<br>
 * <code>f(x) = Sigma<sub>i=0,1,...</sub>(a<sup>i</sup> * cos(b<sup>i</sup> *
 * pi * x))</code><br>
 * However, for multidimensional spaces we extend this function to:<br>
 * <code>f<sup>*</sup>(x) = 1/n * Sigma<sub>i=1..n</sub> f(x<sub>i</sub>)</code>
 * </p>
 * <p>G. H. Hardy proved in <i>Weierstrass non-differentiable function</i> that
 * this function is non-differentiable in any point if a * b &gt;= 1.</p>
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvWeierstrassFunction implements
    EvObjectiveFunction<EvRealVectorIndividual> {

  /** Class serial version. */
  private static final long serialVersionUID = 8829617042081411850L;

  /** Elements of the function's sum (parameter k). */
  private final int depth;
  /** Parameter a of the function. */
  private final double a;
  /** Parameter b of the function. */
  private final double b;

  /**
   * Constructs new instance of the Weierstrass function.
   * 
   * @param depth_ Number of added elements (see parameter k in function class 
   *               description), at least 1 
   * @param param_a Parameter <code>a</code> of the function, must lay between
   *                0.0 and 1.0
   * @param param_b Parameter <code>b</code> of the function, must satisfy
   *                <code>a * b &gt;= 1</code> 
   */
  public EvWeierstrassFunction(final int depth_, final double param_a,
      final double param_b) {
    if (depth_ < 1) {
      throw new IllegalArgumentException("Depth must be positive");
    }
    if (param_a <= 0.0 || param_a >= 1.0) {
      throw new IllegalArgumentException(
          "Parameter 'a' must lay between 0.0 and 1.0, both exclusive");
    }
    if (param_b * param_a < 1.0) {
      throw new IllegalArgumentException("Product of parameters 'a' and 'b'"
          + " must be greater or equal to 1.0");
    }
    
    this.depth = depth_;
    this.a = param_a;
    this.b = param_b;
  }
  
  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvRealVectorIndividual individual) {
    double sum = 0.0;
    for (int i = 0; i < individual.getDimension(); i++) {
      double ith_val = individual.getValue(i);
      double next_a = 1.0;
      double next_b = 1.0;
      double next_sum = 0.0; 
      for (int d = 0; d <= depth; d++) {
        next_sum += next_a * Math.cos(Math.PI * next_b * ith_val);
        next_a *= this.a;
        next_b *= this.b;
      }

      sum += next_sum;
    }

    return sum / individual.getDimension();
  }

}
