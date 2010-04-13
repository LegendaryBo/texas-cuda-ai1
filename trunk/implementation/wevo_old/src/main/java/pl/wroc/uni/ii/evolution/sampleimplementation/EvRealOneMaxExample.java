package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvIterateCompositon;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorAverageCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorDifferentialEvolutionOperator;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorStandardDeviationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

public class EvRealOneMaxExample {

  /**
   * Example of use OneMax on RealVectors
   * 
   * @param <code>args</code> no default parameters
   */
  public static void main(String[] args) {

    /* Some kind of SGA for real numbers */
    EvTask evolutionary_task = new EvTask();
    EvAlgorithm<EvRealVectorIndividual> genericEA =
        new EvAlgorithm<EvRealVectorIndividual>(1000);
    genericEA.setSolutionSpace(new EvRealVectorSpace(
        new EvRealOneMax<EvRealVectorIndividual>(), 3));
    genericEA.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());

    genericEA
        .addOperatorToEnd(new EvReplacementComposition<EvRealVectorIndividual>(
            new EvIterateCompositon<EvRealVectorIndividual>(
                new EvApplyOnSelectionComposition<EvRealVectorIndividual>(
                    new EvTwoSelectionComposition<EvRealVectorIndividual>(
                        new EvRandomSelection<EvRealVectorIndividual>(16, false),
                        new EvKBestSelection<EvRealVectorIndividual>(4)),
                    new EvTwoOperatorsComposition<EvRealVectorIndividual>(
                        new EvRealVectorStandardDeviationMutation(0.01),
                        new EvRealVectorAverageCrossover(4)))),
            new EvBestFromUnionReplacement<EvRealVectorIndividual>()));

    genericEA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvRealVectorIndividual>(
            System.out));

    genericEA
        .setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(20));
    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    System.out.println("Differential evolution");
    /* With differential crossover and tournament selection */
    genericEA = new EvAlgorithm<EvRealVectorIndividual>(1000);
    genericEA.setSolutionSpace(new EvRealVectorSpace(
        new EvRealOneMax<EvRealVectorIndividual>(), 3));
    genericEA.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    genericEA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvRealVectorIndividual>(
            System.out));
    genericEA.addOperatorToEnd(new EvRealVectorDifferentialEvolutionOperator(
        0.1));
    genericEA
        .setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(100));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();
  }
}
