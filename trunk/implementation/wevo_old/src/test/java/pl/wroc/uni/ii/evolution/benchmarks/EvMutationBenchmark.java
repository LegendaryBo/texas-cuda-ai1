package pl.wroc.uni.ii.evolution.benchmarks;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * @author Kacper Gorski
 * 
 * mutation benchmark
 * 
 */
public class EvMutationBenchmark {

  public static void main(String[] args) {

    EvTask evolutionary_task = new EvTask();

    EvAlgorithm<EvBinaryVectorIndividual> sga = new EvSGA<EvBinaryVectorIndividual>(
        100, new EvTwoSelectionComposition<EvBinaryVectorIndividual>(
            new EvRandomSelection<EvBinaryVectorIndividual>(16, false),
            new EvKBestSelection<EvBinaryVectorIndividual>(4)),
        new EvTwoOperatorsComposition<EvBinaryVectorIndividual>(
            new EvBinaryVectorNegationMutation(0.015),
            new EvKnaryVectorOnePointCrossover<EvBinaryVectorIndividual>()),
        new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());

    sga.setSolutionSpace(new EvBinaryVectorSpace(new EvOneMax(), 60));

    sga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        400));
    evolutionary_task.setAlgorithm(sga);

    long cur_time = System.currentTimeMillis();
    
    evolutionary_task.run();
    evolutionary_task.printBestResult();
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " + (after_time - cur_time) + " msec");
    System.out.print("best: ");       

  }

}
