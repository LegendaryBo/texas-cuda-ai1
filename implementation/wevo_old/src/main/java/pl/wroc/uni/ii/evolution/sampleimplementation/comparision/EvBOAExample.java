package pl.wroc.uni.ii.evolution.sampleimplementation.comparision;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.Ev3Deceptive;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Boa example used in comparision with Illegal's implementation.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvBOAExample {

  
  /**
   * Disabling constructor.
   */
  protected EvBOAExample() {
    throw new IllegalStateException("Class cannot be instantiated");
  }
  
  /**
   * 
   * Start the algithm and make some charts.
   * 
   * @param args - not used
   */  
  public static void main(final String[] args) {

    int population_size = 3000;
    
    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    
    EvAlgorithm<EvBinaryVectorIndividual> genericEA;
    
    EvBinaryVectorBOAOperator boa = 
      new EvBinaryVectorBOAOperator(4, 20, population_size);   
    
    EvObjectiveFunction<EvBinaryVectorIndividual> objective_function =
        new Ev3Deceptive();

    genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(population_size);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 60));

    genericEA.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));

    genericEA.addOperatorToEnd(
        new EvTournamentSelection<EvBinaryVectorIndividual>(
            4, 1));

    genericEA.addOperator(boa);
    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(
            30));
    
    
    boa.collectBayesianStats(storage);
 
    long cur_time = System.currentTimeMillis();
    genericEA.init();
    genericEA.run();
    // evolutionary_task.printBestResult();
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " + (after_time - cur_time) + " sec");
    
    
    EvBayesianChart.viewStats(storage.getStatistics());
    
    
  }

}
