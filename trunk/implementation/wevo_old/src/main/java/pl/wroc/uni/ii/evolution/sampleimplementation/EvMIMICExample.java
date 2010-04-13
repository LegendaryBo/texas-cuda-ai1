package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorMIMICOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Example of mimic resolving KDeceptive problem.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvMIMICExample {

  /**
   * Disabling constructor.
   */
  protected EvMIMICExample() {
    throw new IllegalStateException("Class cannot be instantiated");
  }
  
  
  /**
   * Runs algorithm and shows some stats.
   * 
   * @param args - not used
   */
  public static void main(final String[] args) {

    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);

    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    
    EvBinaryVectorMIMICOperator mimic = new EvBinaryVectorMIMICOperator(4000);
    
    EvReplacementComposition<EvBinaryVectorIndividual> replacement = 
      new EvReplacementComposition<EvBinaryVectorIndividual>(
        mimic,
        new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());
    
    mimic.collectBayesianStats(storage);
    
    EvTournamentSelection<EvBinaryVectorIndividual> selection = 
      new EvTournamentSelection<EvBinaryVectorIndividual>(16, 4);
    
    EvAlgorithm<EvBinaryVectorIndividual> alg =
        new EvAlgorithm<EvBinaryVectorIndividual>(4000);
    
    alg.setSolutionSpace(
        new EvBinaryVectorSpace(objective_function, 60));
    
    alg.setObjectiveFunction(objective_function);
    
    alg.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(17));
    
    alg.addOperator(replacement);
    alg.addOperator(selection);
    alg.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));


    long cur_time = System.currentTimeMillis();
    alg.init();
    alg.run();
    long after_time = System.currentTimeMillis();
    
    System.out.println("RUN TIME: " 
        + ((after_time - cur_time) / 1000) + " sec");
    
    EvBayesianChart.viewStats(storage.getStatistics());
  }
}
