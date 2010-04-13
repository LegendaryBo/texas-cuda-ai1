package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyGAOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;
import pl.wroc.uni.ii.evolution.solutionspaces.EvSimplifiedMessySpace;

/**
 * Example of MessyGA using EvMessyOperator
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvSimplifiedMessyGAOperatorExample {
  public static void main(String[] args) {

    // parameters of the example
    final int max_iteration = 200;
    final int population_size = 200;
    final int chromosome_length = 50;
    final int max_value_of_gene = 5;
    final int max_checks_number = 30;
    final int selection_individuals = 50;
    final double crossover_probability = 1.0;
    final double jump_mutation_probability = 0.01;
    final double replace_gene_mutation_probability = 0.01;

    // creating the algorithm
    EvAlgorithm<EvSimplifiedMessyIndividual> messyGA =
        new EvAlgorithm<EvSimplifiedMessyIndividual>(population_size);

    EvSimplifiedMessyObjectiveFunction messyFn =
        new EvSimplifiedMessyObjectiveFunction(max_value_of_gene,
            new EvSimplifiedMessyMaxSum(), max_checks_number);

    messyGA.setSolutionSpace(new EvSimplifiedMessySpace(messyFn,
        chromosome_length, max_value_of_gene));

    messyGA.addOperatorToEnd(new EvSimplifiedMessyGAOperator(max_value_of_gene,
        selection_individuals, crossover_probability,
        jump_mutation_probability, replace_gene_mutation_probability));

    messyGA
        .setTerminationCondition(new EvMaxIteration<EvSimplifiedMessyIndividual>(
            max_iteration));

    messyGA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvSimplifiedMessyIndividual>(
            System.out));

    // running evolutionary task
    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(messyGA);

    evolutionary_task.run();

    System.out.println("--------------------------------");
    System.out.println("BEST pattern:" + messyGA.getBestResult());
    System.out.println("BEST found:"
        + messyGA.getBestResult().best_found_without_empty_genes);
    System.out.println("End value:"
        + messyGA.getBestResult().best_found_without_empty_genes
            .getObjectiveFunctionValue());

  }
}
