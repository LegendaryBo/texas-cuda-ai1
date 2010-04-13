package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyAllelicDoubleMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyCutSpliceOperator;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyGenicMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Omega operator
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaOperator implements EvOperator<EvOmegaIndividual> {

  private final EvOperator<EvOmegaIndividual> cut_and_splice;

  private final EvOperator<EvOmegaIndividual> locus_mutation;

  private final EvOperator<EvOmegaIndividual> allelic_mutation;

  private final int population_size;

  private final int problem_size;

  private final double reduction_ratio;

  private final int max_era;

  private final int max_juxt_loop;

  private final double locus_mutation_probability;

  private final double allelic_mutation_probability;

  private final double growing_population_size_factor;


  /**
   * Constructor
   * 
   * @param population_size - size of population inside Omega
   * @param problem_size - problem size
   * @param reduction_ratio - reductio ratio
   * @param max_era - max era
   * @param max_juxtapositional_loop - max juxtapositional loop
   * @param growing_population_size_factor - growing factor
   * @param cut_probability_factor - cut probability
   * @param splice_probability_factor - splice probability
   * @param locus_mutation_probability - locus (genic) mutation probability
   * @param allelic_mutation_probability - allelic mutation probability
   */
  public EvOmegaOperator(int population_size, int problem_size,
      double reduction_ratio, int max_era, int max_juxtapositional_loop,
      double growing_population_size_factor, double cut_probability_factor,
      double splice_probability_factor, double locus_mutation_probability,
      double allelic_mutation_probability) {

    // error checking
    if (reduction_ratio < 0.0 || reduction_ratio > 1.0)
      throw new IllegalArgumentException("Reduction ratio should be in [0;1]");

    if (cut_probability_factor < 0.0 || cut_probability_factor > 1.0)
      throw new IllegalArgumentException("Cut probability should be in [0;1]");

    this.population_size = population_size;
    this.problem_size = problem_size;
    this.reduction_ratio = reduction_ratio;
    this.max_era = max_era;
    this.max_juxt_loop = max_juxtapositional_loop;
    this.allelic_mutation_probability = allelic_mutation_probability;
    this.locus_mutation_probability = locus_mutation_probability;
    this.growing_population_size_factor = growing_population_size_factor;

    locus_mutation =
        new EvMessyGenicMutation<EvOmegaIndividual>(locus_mutation_probability);
    allelic_mutation =
        new EvMessyAllelicDoubleMutation<EvOmegaIndividual>(
            allelic_mutation_probability, false);

    cut_and_splice =
        new EvMessyCutSpliceOperator<EvOmegaIndividual>(cut_probability_factor,
            splice_probability_factor);
  }


  /**
   * Constructor; growing population is set 2
   * 
   * @param population_size - size of population inside Omega
   * @param problem_size - problem size
   * @param reduction_ratio - reductio ratio
   * @param max_era - max era
   * @param max_juxtapositional_loop - max juxtapositional loop
   * @param cut_probability_factor - cut probability
   * @param splice_probability_factor - splice probability
   * @param locus_mutation_probability - locus (genic) mutation probability
   * @param allelic_mutation_probability - allelic mutation probability
   */
  public EvOmegaOperator(int population_size, int problem_size,
      double reduction_ratio, int max_era, int max_juxtapositional_loop,
      double cut_probability_factor, double splice_probability_factor,
      double locus_mutation_probability, double allelic_mutation_probability) {

    this(population_size, problem_size, reduction_ratio, max_era,
        max_juxtapositional_loop, 1.0, cut_probability_factor,
        splice_probability_factor, locus_mutation_probability,
        allelic_mutation_probability);
  }


  /**
   * Generates new population from the given one.
   * 
   * @param population - given population
   * @return new population - population derived from the given one
   */
  public EvPopulation<EvOmegaIndividual> apply(
      EvPopulation<EvOmegaIndividual> population) {
    // error checking
    if (population == null) {
      throw new IllegalArgumentException("Applied population cannot be null");
    }

    if (population.size() == 0) {
      throw new IllegalArgumentException(
          "Applied population must contain at leat one individual");
    }

    EvPopulation<EvOmegaIndividual> omega_population = population;
    EvOmegaIndividual template = omega_population.getBestResult().toTemplate();
    EvObjectiveFunction<EvOmegaIndividual> obj =
        population.get(0).getObjectiveFunction();

    EvPopulation<EvOmegaIndividual> sub_populations[] =
        new EvPopulation[max_era];
    for (int era = 0; era < max_era; era++) {
      double pop_size = population_size;
      for (int generation = 0; generation <= era; generation++) {
        EvOperator<EvOmegaIndividual> length_reduction =
            new EvOmegaLengthReductionOperator(reduction_ratio, generation + 1);
        sub_populations[generation] =
            this.initPopulation((int) pop_size, problem_size, problem_size,
                obj, template);
        sub_populations[generation] =
            length_reduction.apply(sub_populations[generation]);
        pop_size *= growing_population_size_factor;
      }

      // we merge subpopulations into one big population
      omega_population = merge(sub_populations);

      EvOmegaIndividual best = omega_population.getBestResult();
      for (int juxt_loop = 0; juxt_loop < max_juxt_loop; juxt_loop++) {
        // juxtapositional phase
        omega_population = cut_and_splice.apply(omega_population);

        if (locus_mutation_probability > 0.0) {
          locus_mutation.apply(omega_population);
        }
        if (allelic_mutation_probability > 0.0) {
          allelic_mutation.apply(omega_population);
        }

        EvOmegaIndividual temp = omega_population.getBestResult();
        if (best.compareTo(temp) < 0) {
          best = temp;
        }
      }

      if (template.compareTo(best) < 0) {
        template = best.toTemplate();
      }
    }

    template = makeTemplate(omega_population, template);
    int output_population_size = population.size();
    EvPopulation<EvOmegaIndividual> output =
        population.kBest(output_population_size);
    output.remove(output.getWorstResult());
    output.add(template);

    return output;
  }


  /**
   * Creates population of well-specified omega individuals
   * 
   * @param pop_size - size of initialized population
   * @param genotype_length - length of individual's genotype
   * @param chromosome_length - length of individual's chromosome
   * @param obj - omega objective function
   * @param template - template
   * @return population of omega individuals
   */
  private EvPopulation<EvOmegaIndividual> initPopulation(int pop_size,
      int genotype_length, int chromosome_length,
      EvObjectiveFunction<EvOmegaIndividual> obj, EvOmegaIndividual template) {

    EvOmegaIndividual[] individuals = new EvOmegaIndividual[pop_size];
    for (int i = 0; i < pop_size; i++) {
      individuals[i] =
          new EvOmegaIndividual(genotype_length, template, chromosome_length);
      individuals[i].setObjectiveFunction(obj);
    }

    return new EvPopulation<EvOmegaIndividual>(individuals);
  }


  /**
   * Creates template
   * 
   * @param population
   * @param current_template
   * @return new template
   */
  private EvOmegaIndividual makeTemplate(
      EvPopulation<EvOmegaIndividual> population,
      EvOmegaIndividual current_template) {

    // we choose best individual from population as a template
    EvOmegaIndividual temp = population.getBestResult().toTemplate();
    if (current_template == null)
      current_template = temp;
    else {
      if (current_template.compareTo(temp) < 0) {
        current_template = temp;
      }
    }

    return current_template;
  }


  /**
   * Merges several populations.
   * 
   * @param populations - populations to merge
   * @return merged population
   */
  private EvPopulation<EvOmegaIndividual> merge(
      EvPopulation<EvOmegaIndividual> populations[]) {

    EvPopulation<EvOmegaIndividual> pop = new EvPopulation<EvOmegaIndividual>();
    int pop_size = populations.length;
    for (int i = 0; i < pop_size; i++) {
      if (populations[i] != null) {
        pop.addAll(populations[i]);
      }
    }
    return pop;
  }
}