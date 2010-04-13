package pl.wroc.uni.ii.evolution.benchmarks;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * @author Kacper Gorski
 *
 */
public class EvUniformCrossoverBenchmark {

  public static void main(String[] args) {

    EvOneMax objective_function = new EvOneMax();
    
    EvTask evolutionary_task = new EvTask();
    
    EvAlgorithm<EvBinaryVectorIndividual> genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(100);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function,60));
    genericEA.setObjectiveFunction(objective_function);
    genericEA.addOperatorToEnd(new EvKBestSelection<EvBinaryVectorIndividual>(50));
    genericEA.addOperatorToEnd(
        new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>());
    genericEA.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(500));

    long cur_time = System.currentTimeMillis();    
    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();  
    evolutionary_task.printBestResult();
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " + (after_time - cur_time) + " msec");
    System.out.print("best: ");       

  }

}
