package pl.wroc.uni.ii.evolution.benchmarks;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvPositionSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * @author Kacper Gorski
 *
 */
public class EvPositionSelectionBenchmark {


  public static void main(String[] args) {

    int pop_size=1000;
    
    EvTask evolutionary_task = new EvTask();

    EvOneMax objective_function = new EvOneMax();
   
    EvAlgorithm<EvBinaryVectorIndividual> alg = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
     
    alg.addOperatorToEnd(new EvPositionSelection<EvBinaryVectorIndividual>(500));

    alg.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(1000));    

    alg.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 60));    

    evolutionary_task.setAlgorithm(alg);
    long cur_time = System.currentTimeMillis();
    evolutionary_task.run();
    evolutionary_task.printBestResult();
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " + (after_time - cur_time) + " msec");
    System.out.print("best: ");      
    
  }  
  
  
}
