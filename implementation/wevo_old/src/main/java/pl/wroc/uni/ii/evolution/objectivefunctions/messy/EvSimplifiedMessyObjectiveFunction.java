package pl.wroc.uni.ii.evolution.objectivefunctions.messy;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Wrapper class for evaluating objective functions on MessyIndividual class
 * instances.
 * 
 * @author Krzysztof Sroka, Marcin Golebiowski, Kacper Gorski
 */
public final class EvSimplifiedMessyObjectiveFunction implements
    EvObjectiveFunction<EvSimplifiedMessyIndividual> {

  private static final long serialVersionUID = -3558021101397905305L;

  private final int max_value_of_gene;

  private final EvObjectiveFunction<EvSimplifiedMessyIndividual> function;

  private int max_checks_number;


  /**
   * Constructor.
   * 
   * @param max_value_of_gene Maximum value of gene
   * @param fun Objective function to evaluate
   * @param max_checks_number
   */
  public EvSimplifiedMessyObjectiveFunction(int max_value_of_gene,
      EvObjectiveFunction<EvSimplifiedMessyIndividual> fun,
      int max_checks_number) {

    if (max_value_of_gene <= 0) {
      throw new NullPointerException(
          "Maximum value of gene must be higher than 0");
    }

    if (fun == null) {
      throw new NullPointerException("Objective function is null");
    }

    this.max_checks_number = max_checks_number;
    this.function = fun;
    this.max_value_of_gene = max_value_of_gene;

  }


  /**
   * Evaluates objective function for random individuals derived from given
   * individual. Returns objective value of best found individual. Found
   * individual is stored in given individual
   * 
   * @param individual MessyIndividual to evaulate. <BR>
   *        It must belong to MessySpace parameter from contructor
   * @return Biggest possible sum made from individual
   */
  public double evaluate(EvSimplifiedMessyIndividual individual) {
    EvSimplifiedMessyIndividual ind = getBest(individual);
    individual.best_found_without_empty_genes = ind;

    return function.evaluate(ind);
  }


  /**
   * Evaluates given objective function for random individuals derived from
   * given individual. Returns best individual found.
   * 
   * @param individual MessyIndividual to evaulate. <BR>
   *        It must belong to MessySpace parameter from contructor
   * @return individual with biggest possible sum
   */
  private EvSimplifiedMessyIndividual getBest(
      EvSimplifiedMessyIndividual individual) {

    EvSimplifiedMessyIndividual best = null;
    double best_value = Double.NEGATIVE_INFINITY;

    /**
     * Check only max_checks_number random derivaties of individual
     */
    for (int i = 0; i < max_checks_number; i++) {

      EvSimplifiedMessyIndividual random_ind =
          getRandomDerive(individual, max_value_of_gene);

      double ind_value = function.evaluate(random_ind);
      if (ind_value > best_value) {
        best = random_ind;
        best_value = ind_value;
      }
    }
    return best;
  }


  private EvSimplifiedMessyIndividual getRandomDerive(
      EvSimplifiedMessyIndividual individual, int max_gene_value) {
    EvSimplifiedMessyIndividual new_ind =
        new EvSimplifiedMessyIndividual(individual.getLength());

    new_ind.setObjectiveFunction(individual.getObjectiveFunction());

    for (int i = 0; i < new_ind.getLength(); i++) {
      ArrayList<Integer> gene_values = individual.getGeneValues(i);

      if (gene_values.size() == 0) {
        new_ind.setGeneValue(i, EvRandomizer.INSTANCE.nextInt(0,
            max_gene_value, true));
      } else {
        new_ind.setGeneValue(i, gene_values.get(EvRandomizer.INSTANCE
            .nextInt(gene_values.size())));
      }
    }

    return new_ind;
  }
}
