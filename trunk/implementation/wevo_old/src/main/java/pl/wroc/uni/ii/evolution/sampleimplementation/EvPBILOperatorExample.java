package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorPBILOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * simple example of usage of PBIL operator.
 * 
 * @author Kacper Gorski
 */
public class EvPBILOperatorExample {

  public static void main(String[] args) {
    final int pop_size = 100;
    final int bits = 150;

    EvAlgorithm<EvBinaryVectorIndividual> alg =
        new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    EvOneMax one_max = new EvOneMax();
    EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
    EvBinaryVectorPBILOperator pbil =
        new EvBinaryVectorPBILOperator(one_max, 10, 0.03, 0.02, 0.02, bits,
            solution_space);
    alg.setObjectiveFunction(one_max);
    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(pbil);
    alg
        .setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
            30));
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
