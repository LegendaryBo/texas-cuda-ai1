package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKPattern;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Simple test of BOA operator.
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

    EvTask evolutionary_task = new EvTask();

    EvAlgorithm<EvBinaryVectorIndividual> genericEA;
 
    EvBinaryVectorBOAOperator boa = new
      EvBinaryVectorBOAOperator(4, 20, 4000);
    
    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
        
    boa.collectBayesianStats(storage);
        
    EvKPattern objective_function = new EvKPattern(new int[] {1, 0, 1, 0}, 4);

    genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(2000);
    genericEA.setSolutionSpace(
        new EvBinaryVectorSpace(objective_function, 40));

    genericEA.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));

    genericEA.addOperatorToEnd(
        new EvTournamentSelection<EvBinaryVectorIndividual>(
            20, 8));

    genericEA.addOperatorToEnd(
        new EvReplacementComposition<EvBinaryVectorIndividual>(
            boa,
            new EvBestFromUnionReplacement<EvBinaryVectorIndividual>()));

    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(
            15));

    evolutionary_task.setAlgorithm(genericEA);
    long cur_time = System.currentTimeMillis();
    evolutionary_task.run();
    evolutionary_task.printBestResult();
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " 
        + ((after_time - cur_time) / 1000) + " sec");

    
    EvBayesianChart.viewStats(storage.getStatistics());
    
    
  }

}
