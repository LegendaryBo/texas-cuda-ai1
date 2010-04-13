package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyGeneDeletionMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;

/**
 * Length reduction operator (useful in omega); <br/> Assumption: Every
 * individual in population has the same size.
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaLengthReductionOperator implements
    EvOperator<EvOmegaIndividual> {

  private final double reduction_ratio;

  private final int lower_bound_for_expected_ind_len;


  public EvOmegaLengthReductionOperator(double reduction_ratio,
      int lower_bound_for_expected_ind_length) {

    this.reduction_ratio = reduction_ratio;
    this.lower_bound_for_expected_ind_len = lower_bound_for_expected_ind_length;
  }


  public EvPopulation<EvOmegaIndividual> apply(
      EvPopulation<EvOmegaIndividual> population) {

    int pop_size = population.size();
    int expected_len = lower_bound_for_expected_ind_len;

    EvPopulation<EvOmegaIndividual> omega_population = population.clone();
    int problem_size = omega_population.get(0).getGenotypeLength();
    final EvOperator<EvOmegaIndividual> omega_selection =
        new EvOmegaSelection(pop_size, problem_size);
    int one_ind_len = omega_population.get(0).getChromosomeLength();
    int next_ind_len = (int) Math.floor(one_ind_len * reduction_ratio);

    while (next_ind_len > expected_len) {
      int ns = computeNS(one_ind_len, next_ind_len, expected_len);
      int nd = one_ind_len - next_ind_len;
      for (int i = 0; i < ns; i++) {
        omega_population = omega_selection.apply(omega_population);
      }
      final EvMutation<EvOmegaIndividual> gene_deletion =
          new EvMessyGeneDeletionMutation<EvOmegaIndividual>(nd, false);

      omega_population = gene_deletion.apply(omega_population);
      one_ind_len = next_ind_len;
      next_ind_len = (int) Math.floor(one_ind_len * reduction_ratio);
    }

    if (one_ind_len > expected_len) {
      int ns = computeNS(one_ind_len, expected_len, expected_len);
      for (int i = 0; i < ns; i++) {
        omega_population = omega_selection.apply(omega_population);
      }
      final EvMutation<EvOmegaIndividual> gene_deletion =
          new EvMessyGeneDeletionMutation<EvOmegaIndividual>(one_ind_len
              - expected_len, false);
      omega_population = gene_deletion.apply(omega_population);
    }

    return omega_population;
  }


  /**
   * Compute number of individuals to select TODO compute it in other way
   * 
   * @param current_int_len
   * @param next_ind_len
   * @param number_of_era
   * @param population_size
   * @return
   */
  private final int computeNS(int current_int_len, int next_ind_len,
      int number_of_era) {

    long numerator = fact(current_int_len, number_of_era);
    long denominator = fact(next_ind_len, number_of_era);
    double q = numerator / denominator;
    if (q <= 1.0) {
      return 1;
    } else {
      double log2 = Math.log10(q) / Math.log10(2);
      return (int) Math.ceil(log2);
    }
  }


  /**
   * @param n
   * @param k
   * @return n(n-1)...(n-k+1)
   */
  private long fact(int n, int k) {
    long prod = 1;
    for (int i = n; i > n - k; i--) {
      prod *= i;
    }

    return prod;
  }
}