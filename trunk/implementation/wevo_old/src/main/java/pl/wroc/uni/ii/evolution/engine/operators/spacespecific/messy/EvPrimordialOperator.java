package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * The Messy Primordial Phase Operator. This operator implements first phase of
 * Messy Genetic Algorithm. It usually works with large population of short
 * individuals (building blocks) and try to reduce number of them by selecting
 * better ones. It uses a pairs tournament selection (with optional thesholding
 * and tie breaking) to control population size and selecting better
 * individuals.
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvPrimordialOperator<T extends EvMessyIndividual> implements
    EvOperator<T> {

  // Parameters dependent of the algorithm

  /* Size of the problem, length of genotype */
  private int problem_size;

  /* If thresholding is desired */
  private boolean thresholding;

  /* If tiebreaking is desired */
  private boolean tie_breaking;

  /*
   * This array contains the number of duplicates of each individual in the
   * initial population for each era
   */
  private int[] copies;

  /* Population size in the juxtapositional phase for each era */
  private int[] juxtapositional_sizes;

  // Parameters dependent of the moment

  /* Current era */
  private int era = 0;

  /* Current generation */
  private int generation = 1;

  /*
   * It is set to true, if the population size needs to be reduced at current
   * generation during reduction at every other generation
   */
  private boolean cut_population = false;

  // Parameters calculated by the primordial phase

  /* The duration of the primordial phase */
  private int primordial_generation;

  /* The duration of the population reduction in the primordial phase */
  private int cut_generation;

  /*
   * The duration for population reduction at every other generation. Beyond
   * this generation, the population is reduced at every generation until
   * cut_generation generation
   */
  private int cut_every_other_generation;

  /* Objective function value of the template */
  private double templatefitness;


  /**
   * Constructor creates the primordial operator with options.
   * 
   * @param problem_size - size of the problem, length of genotype
   * @param thresholding - true if thresholding is desired
   * @param tie_breaking - true if tiebreaking is desired
   * @param copies - the number of duplicates of each individual in the initial
   *        population for each era
   * @param juxtapositional_sizes - population sizes in the juxtapositional
   *        phase for each era, it means population size after primordial phase.
   */
  public EvPrimordialOperator(int problem_size, boolean thresholding,
      boolean tie_breaking, int[] copies, int[] juxtapositional_sizes) {

    this.problem_size = problem_size;
    this.thresholding = thresholding;
    this.tie_breaking = tie_breaking;
    this.copies = copies;
    this.juxtapositional_sizes = juxtapositional_sizes;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {

    int population_size = changePopulationSize(population.size());

    EvMessyPairsTournamentSelection<T> selection =
        new EvMessyPairsTournamentSelection<T>(population_size, thresholding,
            tie_breaking);

    EvPopulation<T> new_population = selection.apply(population);

    generation++;
    return new_population;
  }


  /**
   * Update parameters for a new era.
   * 
   * @param population - new generated population for the new era
   * @param templatefitness - objective function value of the template
   * @return number of iterations of primordial phase which should be applied
   */
  public int setupNewEra(EvPopulation<T> population, double templatefitness) {

    this.templatefitness = templatefitness;
    era++;
    generation = 1;
    cut_population = false;
    setupPopulationSize(population);

    return primordial_generation;
  }


  /*
   * Changes population size for current generation. @param population_size
   * -current size of the population @return new size for the population
   */
  private int changePopulationSize(int population_size) {

    if (generation <= cut_generation) {

      if ((generation == cut_generation)
          && (population_size > juxtapositional_sizes[era - 1]))
        population_size = juxtapositional_sizes[era - 1];
      else if (generation <= cut_every_other_generation) {
        // Cut at every other generation
        if (cut_population) {
          population_size = population_size / 2;
          cut_population = false;
        } else
          cut_population = true;
      } else
        // Cut at every generation
        population_size = population_size / 2;

    }
    return (population_size);
  }


  /*
   * Sets up the population size control parameter in primordial phase. @param
   * population -population for which primordial phase will work
   */
  private void setupPopulationSize(EvPopulation<T> population) {

    int population_size = population.size();
    if (population_size < 1)
      throw new IllegalStateException("Population size must by greater then 0");

    if (population_size < juxtapositional_sizes[era - 1])
      juxtapositional_sizes[era - 1] = population_size;

    // Initial proportion of the best individual
    double proportion_1 =
        (double) (problem_size * copies[era - 1])
            / (double) (era * population_size);
    double proportion_2 = getProportionBetterIndividuals(population);
    // Choose the greater of two
    double proportion =
        (proportion_1 > proportion_2) ? proportion_1 : proportion_2;

    // The duration of primordial phase: (under tournament selection) is
    // calculated to have the proportion of best individual equals to 0.5
    primordial_generation = 0;
    if (proportion < 0.5)
      primordial_generation =
          (int) Math.floor(Math.log((1.0 - proportion) / proportion)
              / Math.log(2.0));

    // Maximum number of cut
    int maximum_cut = primordial_generation;

    // Calculate number of cut ~ log_2 (population_size / juxtapositional_size)
    int number_of_cut =
        (int) Math.round((Math.log(population_size) - Math
            .log(juxtapositional_sizes[era - 1]))
            / Math.log(2.0));

    if (maximum_cut <= 0) {
      // If no cut is required
      cut_generation = 0;
      juxtapositional_sizes[era - 1] = population_size;
    } else if (number_of_cut > maximum_cut) {
      // Too many cut is requested.
      // Given juxtapositional population size is too small, using it instead.
      juxtapositional_sizes[era - 1] =
          (int) Math.round(population_size / (Math.pow(2, maximum_cut)));
      cut_every_other_generation = 0;
      cut_generation = primordial_generation;
    } else if (number_of_cut >= maximum_cut / 2) {
      // Some cut needs to be at every generation
      cut_every_other_generation = 2 * (maximum_cut - number_of_cut) - 1;
      cut_generation = primordial_generation;
    } else {
      // Only cut at every other generation =
      cut_every_other_generation = 2 * number_of_cut - 1;
      cut_generation = cut_every_other_generation;
    }
  }


  /*
   * Get proportion of better individuals than template in the population.
   * @param population - population to calculate proportion @return proportion
   * of better individuals than template.
   */
  private double getProportionBetterIndividuals(EvPopulation<T> population) {
    int count = 0;
    for (int i = 0; i < population.size(); i++)
      if (population.get(i).getObjectiveFunctionValue() >= templatefitness)
        count++;
    return (double) count / population.size();
  }

}