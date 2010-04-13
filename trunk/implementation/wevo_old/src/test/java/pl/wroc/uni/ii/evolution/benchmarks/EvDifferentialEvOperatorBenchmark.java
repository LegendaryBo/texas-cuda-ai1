package pl.wroc.uni.ii.evolution.benchmarks;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorDifferentialEvolutionOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * 
 * @author Kacper Gorski
 *
 */
public class EvDifferentialEvOperatorBenchmark {


  public static void main(String[] args) {
    
    EvTask evolutionary_task = new EvTask();
    EvAlgorithm<EvRealVectorIndividual> genericEA = new EvAlgorithm<EvRealVectorIndividual>(1000);    
    genericEA = new EvAlgorithm<EvRealVectorIndividual>(1000);
    genericEA.setSolutionSpace(new EvRealVectorSpace(new EvRealOneMax<EvRealVectorIndividual>(), 3));
    genericEA.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    genericEA.addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvRealVectorIndividual>(System.out));
    genericEA.addOperatorToEnd(new EvRealVectorDifferentialEvolutionOperator(0.1));
    genericEA.setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(100));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

  }

}
