package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;

/**
 * Class implementing MessyGA algorithm as operator. This class uses four
 * operators: EvRouletteSelection, EvMessyCrossover, EvMessyJumpMutation,
 * EvMessyReplaceGeneMutation and replacement EvBestFromUnionReplacement.
 * 
 * @author Piotr Staszak (stachhh@gmail.com)
 * @author Marek Szykula (marek.esz@gmail.com)
 */

public class EvSimplifiedMessyGAOperator implements
    EvOperator<EvSimplifiedMessyIndividual> {

  private EvOperator<EvSimplifiedMessyIndividual> roulette_selection;

  private EvOperator<EvSimplifiedMessyIndividual> crossover;

  private EvOperator<EvSimplifiedMessyIndividual> jump_mutation;

  private EvOperator<EvSimplifiedMessyIndividual> replace_gene_mutation;

  private EvReplacement<EvSimplifiedMessyIndividual> replacement;


  /**
   * @param max_value_of_gene - maximum value of gene
   * @param selection_individuals - number of invidiual to select
   * @param crossover_probability - probability of crossover
   * @param jump_mutation_probability - probability of jump mutation
   * @param replace_gene_mutation_probability - probability of replace gene
   *        mutation
   */
  public EvSimplifiedMessyGAOperator(int max_value_of_gene,
      int selection_individuals, double crossover_probability,
      double jump_mutation_probability, double replace_gene_mutation_probability) {

    // check if arguments are ok
    if (max_value_of_gene <= 0) {
      throw new IllegalArgumentException(
          "Maximum value of gene must be a positive Integer");
    }

    if (selection_individuals <= 0) {
      throw new IllegalArgumentException(
          "Selection individuals number must be a larger than 0");
    }

    if (crossover_probability < 0.0 || crossover_probability > 1.0) {
      throw new IllegalArgumentException(
          "Crossover probability must be a Double in range[0,1]");
    }

    if (jump_mutation_probability < 0.0 || jump_mutation_probability > 1.0) {
      throw new IllegalArgumentException(
          "Jump mutation probability must be a Double in range[0,1]");
    }

    if (replace_gene_mutation_probability < 0.0
        || replace_gene_mutation_probability > 1.0) {
      throw new IllegalArgumentException(
          "Replace gene mutation probability must be a Double in range[0,1]");
    }

    // initializing operators used by this operator
    roulette_selection =
        new EvRouletteSelection<EvSimplifiedMessyIndividual>(
            new EvIndividualFitness<EvSimplifiedMessyIndividual>(),
            selection_individuals);

    crossover = new EvSimplifiedMessyCrossover(crossover_probability);

    jump_mutation =
        new EvSimplifiedMessyJumpMutation(jump_mutation_probability);

    replace_gene_mutation =
        new EvSimplifiedMessyReplaceGeneMutation(
            replace_gene_mutation_probability, max_value_of_gene);

    replacement = new EvBestFromUnionReplacement<EvSimplifiedMessyIndividual>();
  }


  /**
   * generate new population from the given one
   * 
   * @param population - given population
   */
  public EvPopulation<EvSimplifiedMessyIndividual> apply(
      EvPopulation<EvSimplifiedMessyIndividual> population) {

    // error checking
    if (population == null) {
      throw new IllegalArgumentException("Applied population cannot be null");
    }

    if (population.size() == 0) {
      throw new IllegalArgumentException(
          "Applied population must contain at leat one individual");
    }

    // apply the operators and replacement
    // created during initialization to given population
    EvPopulation<EvSimplifiedMessyIndividual> populationAfterSelection =
        roulette_selection.apply(population);

    EvPopulation<EvSimplifiedMessyIndividual> populationAfterCrossover =
        crossover.apply(populationAfterSelection);

    EvPopulation<EvSimplifiedMessyIndividual> populationAfterJumpMutation =
        jump_mutation.apply(populationAfterCrossover);

    EvPopulation<EvSimplifiedMessyIndividual> populationAfterReplaceGeneMutation =
        replace_gene_mutation.apply(populationAfterJumpMutation);

    EvPopulation<EvSimplifiedMessyIndividual> populationAfterReplacement =
        replacement.apply(population, populationAfterReplaceGeneMutation);

    return populationAfterReplacement;
  }
}
