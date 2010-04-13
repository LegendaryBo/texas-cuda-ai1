package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMGAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvMessyBinaryVectorObjectiveFunctionWrapper;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMessyBinaryVectorSpace;

/**
 * An example of using the Messy Genetic Algorithm Operator, it solves
 * kDeceptiveOneMax for k = 6 and 60 vector length.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */
public class EvMGAOperatorExample {
  public static void main(String[] args) {

    // Parameters of the example

    /* Number of eras */
    final int maximum_era = 3;
    /* Size of the problem, length of vector */
    final int problem_size = 60;
    /*
     * Upper limit of the population size, 0 means that there is no limit. If
     * generated cover individuals number exceedes this, an uniformly random
     * individuals will be selected from them, this option can preserve from out
     * of memory due to extreme big populations, so it enables using more eras,
     * NOTE: this option does not belong to the original mGA
     */
    final int maximum_population_size = 100000;
    /*
     * Probability of cut an individual, multiplied by the length of the
     * chromosome, recommended 1.0 / (2 * problem_size) value
     */
    final double probability_of_cut = 1.0 / (2 * problem_size);
    /* Probability of splice two individuals, recommended high values or 1.0 */
    final double probability_of_splice = 1.0;
    /*
     * Probability of allele negation for each allele, recommended small or 0.0
     */
    final double probability_of_allelic_mutation = 0.0;
    /*
     * Probability of change gene which allele belongs, recommended small or
     * 0.0, NOTE: this not guarantying changing gene to a different one, for
     * probability guarantying changing gene use genic_mutation =
     * changing_genic_mutation * (problem_length / (problem_length-1)), in the
     * original mGA guarantying changing gene mutation is used
     */
    final double probability_of_genic_mutation = 0.0;
    /*
     * There will be compared individuals with a number of common expressed
     * genes larger than expected in random chromosomes
     */
    final boolean thresholding = false;
    /*
     * Shorter individuals have advantage when the objective function value is
     * the same
     */
    final boolean tie_breaking = true;
    /*
     * Negated template is used for generated individuals instead all allele
     * combinations.
     */
    final boolean reduced_initial_population = true;
    /*
     * Find and keep for the best individual in whole era time, instead of get
     * it from final era population, NOTE: this option is an experimental
     * extension, it does not belong to the original mGA.
     */
    final boolean keep_era_best_individual = false;
    /*
     * This array contains the number of duplicates of each individual in the
     * initial population for each era
     */
    final int[] copies = new int[] {5, 2, 1};
    /* Number of generations, specified for all eras */
    final int[] maximum_generationes = new int[] {20, 20, 30};
    /* Population size in the juxtapositional phase, specified for each era */
    final int[] juxtapositional_sizes = new int[] {2000, 2000, 2000};

    EvMGAOperator mga_operator =
        new EvMGAOperator(maximum_era, problem_size, maximum_population_size,
            probability_of_cut, probability_of_splice,
            probability_of_allelic_mutation, probability_of_genic_mutation,
            thresholding, tie_breaking, reduced_initial_population,
            keep_era_best_individual, copies, maximum_generationes,
            juxtapositional_sizes);

    // Create the algorithm
    EvAlgorithm<EvMessyBinaryVectorIndividual> messyGA =
        new EvAlgorithm<EvMessyBinaryVectorIndividual>(1);

    EvMessyBinaryVectorObjectiveFunctionWrapper objective_function =
        new EvMessyBinaryVectorObjectiveFunctionWrapper(new EvKDeceptiveOneMax(
            6));

    messyGA.setSolutionSpace(new EvMessyBinaryVectorSpace(objective_function,
        problem_size));

    int iteration_number = 0;
    for (int i = 0; i < maximum_era; i++)
      iteration_number += maximum_generationes[i];
    messyGA
        .setTerminationCondition(new EvMaxIteration<EvMessyBinaryVectorIndividual>(
            iteration_number));

    messyGA.addOperatorToEnd(mga_operator);

    messyGA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvMessyBinaryVectorIndividual>(
            System.out));

    // Run the algorithm
    EvTask task = new EvTask();
    task.setAlgorithm(messyGA);
    long startTime = System.currentTimeMillis();
    task.run();
    long endTime = System.currentTimeMillis();
    System.out.println("Total time: " + ((double) endTime - startTime) / 1000
        + "s");
  }

}