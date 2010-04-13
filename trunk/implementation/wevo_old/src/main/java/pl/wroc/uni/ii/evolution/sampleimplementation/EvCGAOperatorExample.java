package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorCGAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Simple example of CGA operator
 * 
 * @author Kacper Gorski
 */
public class EvCGAOperatorExample {

  public static void main(String[] args) {

    final int pop_size = 2;
    final int bits = 150;

    EvAlgorithm<EvBinaryVectorIndividual> alg =
        new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);

    EvOneMax one_max = new EvOneMax();
    EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
    EvBinaryVectorCGAOperator cga =
        new EvBinaryVectorCGAOperator(100, 0.02, pop_size, solution_space);

    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(cga);
    alg
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));

    alg
        .setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(8));

    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(alg);

    long cur_time = System.currentTimeMillis();
    evolutionary_task.run();
    long after_time = System.currentTimeMillis();

    System.out
        .println("RUN TIME: " + ((after_time - cur_time) / 1000) + " sec");
    System.out.println(alg.getBestResult());
  }
}
