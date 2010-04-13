package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyJumpMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyReplaceGeneMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;
import pl.wroc.uni.ii.evolution.solutionspaces.EvSimplifiedMessySpace;

/**
 * Example evolutionary algorithm using MessySpace
 * 
 * @author Marcin Golebiewski, Krzysztof Sroka
 */

public class EvSimplifiedMessyExample {
  public static void main(String[] args) {

    EvTask evolutionary_task = new EvTask();
    EvAlgorithm<EvSimplifiedMessyIndividual> messyGA =
        new EvSGA<EvSimplifiedMessyIndividual>(200,
            new EvRouletteSelection<EvSimplifiedMessyIndividual>(
                new EvIndividualFitness<EvSimplifiedMessyIndividual>(), 50),
            new EvTwoOperatorsComposition<EvSimplifiedMessyIndividual>(
                new EvSimplifiedMessyReplaceGeneMutation(0.01, 5),
                new EvTwoOperatorsComposition<EvSimplifiedMessyIndividual>(
                    new EvSimplifiedMessyJumpMutation(0.01),
                    new EvSimplifiedMessyCrossover(1.0))),
            new EvBestFromUnionReplacement<EvSimplifiedMessyIndividual>());

    EvSimplifiedMessyObjectiveFunction messyFn =
        new EvSimplifiedMessyObjectiveFunction(5,
            new EvSimplifiedMessyMaxSum(), 30);
    messyGA.setSolutionSpace(new EvSimplifiedMessySpace(messyFn, 50, 5));
    messyGA
        .setTerminationCondition(new EvMaxIteration<EvSimplifiedMessyIndividual>(
            200));

    messyGA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvSimplifiedMessyIndividual>(
            System.out));

    evolutionary_task.setAlgorithm(messyGA);
    evolutionary_task.run();

    System.out.println("--------------------------------");
    System.out.println("BEST pattern:"
        + (EvSimplifiedMessyIndividual) messyGA.getBestResult());
    System.out
        .println("BEST found:"
            + (EvSimplifiedMessyIndividual) messyGA.getBestResult().best_found_without_empty_genes);
    System.out.println("End value:"
        + messyGA.getBestResult().best_found_without_empty_genes
            .getObjectiveFunctionValue());
  }
}
