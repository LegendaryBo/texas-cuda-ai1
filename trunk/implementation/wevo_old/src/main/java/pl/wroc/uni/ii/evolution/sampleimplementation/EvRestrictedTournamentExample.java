package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.likeness.EvHammingDistanceLikenes;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvRestrictedTournamentReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * @author Kacper Gorski (admin@34al.org)
 *
 */
public class EvRestrictedTournamentExample {

    /**
     * Disabling constructor.
     */
    protected EvRestrictedTournamentExample() {
      throw new IllegalStateException("Class shouldn't be instantiated");
    }
    
   /**
    * Starts example.
    * 
    * @param args - not used
    */
    public static void main(final String[] args) {
      
      int vector_size = 120;
      int pop_size = 1000;
      int iterations = 2000;
      
      EvTask evolutionary_task = new EvTask();

      EvAlgorithm<EvBinaryVectorIndividual> genericEA;
   
      EvBinaryVectorBOAOperator boa = new
        EvBinaryVectorBOAOperator(vector_size, 4, pop_size);
      
      EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
          
      boa.collectBayesianStats(storage);
          
      EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(6);

      genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
      genericEA.setSolutionSpace(
          new EvBinaryVectorSpace(objective_function, vector_size));

      genericEA.addOperatorToEnd(
          new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
              System.out));

      genericEA.addOperatorToEnd(
          new EvReplacementComposition<EvBinaryVectorIndividual>(
              boa,
              new EvRestrictedTournamentReplacement<EvBinaryVectorIndividual>(
                  vector_size, new EvHammingDistanceLikenes())));

      genericEA.setTerminationCondition(
          new EvMaxIteration<EvBinaryVectorIndividual>(
              iterations));

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
