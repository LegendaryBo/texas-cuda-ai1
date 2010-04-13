package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import Jama.Matrix;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * It is mutation operator. It can be used for MiLambdaRoKappaIndividual. It
 * rotate given individual according to matrix which is computing in very
 * interesting way (as it is said in description of ES(mi, lambda, ro, kappa)).
 * It has computation complexity O(n^5). It change only first individual in
 * population (which is correct for MiLambdaRoKappa strategy).
 * 
 * @author Lukasz Witko, Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvMiLambdaRoKappaRotationMutation extends
    EvMutation<EvMiLambdaRoKappaIndividual> {

  private double beta;

  private double tau;

  private double tau_prim;

  private Matrix matrix_epsilon;

  private Matrix matrix_z;


  /**
   * Modificator from tau_prim is draw once for vector, and modificator from tau
   * is draw for each part of vector
   * 
   * @param beta parameter that is used to modification alphas vector
   * @param tau parameter that is used to modification probabilities vector
   * @param tau_prim parameter that is used to modification probabilities vector
   *        (sometimes it is called tau_zero)
   */
  public EvMiLambdaRoKappaRotationMutation(double beta, double tau,
      double tau_prim) {
    this.beta = beta;
    this.tau = tau;
    this.tau_prim = tau_prim;
  }


  @Override
  public EvMiLambdaRoKappaIndividual mutate(
      EvMiLambdaRoKappaIndividual individual) {

    // disturbing alpha values
    for (int i = 0; i < individual.getAlphaLength(); i++) {
      individual.setAlpha(i, individual.getAlpha(i)
          + EvRandomizer.INSTANCE.nextGaussian() * beta);
    }

    // drawing epsilon0
    double epsilon0 = EvRandomizer.INSTANCE.nextGaussian() * tau_prim;

    double[] z = new double[individual.getDimension()];

    // disturbing probabilities vector
    for (int i = 0; i < individual.getDimension(); i++) {
      individual.setProbability(i, individual.getProbability(i)
          * Math.exp(epsilon0 + EvRandomizer.INSTANCE.nextGaussian() * tau));

      z[i] =
          EvRandomizer.INSTANCE.nextGaussian() * individual.getProbability(i);
    }

    int d = individual.getDimension();
    Matrix matrix_t;
    Matrix matrix_tpq;

    matrix_t = Matrix.identity(d, d);
    matrix_z = new Matrix(z, 1);

    matrix_z = matrix_z.transpose();

    int j = 0;
    // computing rotation matrix
    // - this part takes time O(n^5), assuming that we multiply 2 matrices in
    // O(n^3)
    for (int p = 0; p < d - 1; p++) {
      for (int q = p + 1; q < d; q++) {
        j++;
        matrix_tpq = Matrix.identity(d, d);
        matrix_tpq.set(p, p, Math.cos(individual.getAlpha(j)));
        matrix_tpq.set(p, q, -Math.sin(individual.getAlpha(j)));
        matrix_tpq.set(q, p, Math.sin(individual.getAlpha(j)));
        matrix_tpq.set(q, q, Math.cos(individual.getAlpha(j)));

        matrix_t = matrix_t.times(matrix_tpq);
      }
    }

    matrix_epsilon = matrix_t.times(matrix_z);

    // doing rotation of individual
    for (int i = 0; i < d; i++) {
      individual.setValue(i, individual.getValue(i) + matrix_epsilon.get(i, 0));
    }

    return individual;
  }
}
