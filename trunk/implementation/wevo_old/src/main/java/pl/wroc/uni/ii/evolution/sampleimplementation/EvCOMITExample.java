package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorCOMITOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Example of mimic resolving deceptive problem.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvCOMITExample {

  /**
   * Disabling constructor.
   */
  protected EvCOMITExample() {
    throw new IllegalStateException("Class cannot be instantiated");
  }
  
  /**
   * @param args - not used
   */
  public static void main(final String[] args) {

    int pop_size = 4000;
    int bits = 60;
    
    EvAlgorithm<EvBinaryVectorIndividual> alg = 
        new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    EvKDeceptiveOneMax one_max = new EvKDeceptiveOneMax(4);
    EvBinaryVectorSpace solution_space = 
        new EvBinaryVectorSpace(one_max, bits);
    EvBinaryVectorCOMITOperator comit = 
        new EvBinaryVectorCOMITOperator(pop_size);
    EvReplacementComposition<EvBinaryVectorIndividual> replacement = 
      new EvReplacementComposition<EvBinaryVectorIndividual>(
          comit,
        new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());    
    
    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    comit.collectBayesianStats(storage);
    
    //EvTournamentSelection<EvBinaryVectorIndividual> selection = 
    //  new EvTournamentSelection<EvBinaryVectorIndividual>(16,4);
    EvRouletteSelection<EvBinaryVectorIndividual> selection = 
      new EvRouletteSelection<EvBinaryVectorIndividual>(
          new EvIndividualFitness<EvBinaryVectorIndividual>(), pop_size);
    alg.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(100));
    alg.setObjectiveFunction(one_max);
    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(replacement);
    alg.addOperatorToEnd(selection);
    alg.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    
    long cur_time = System.currentTimeMillis();    
    alg.init();
    alg.run();
    long after_time = System.currentTimeMillis();
    
    System.out.println("RUN TIME: " + (after_time - cur_time) + "mili sec");
    
    EvBayesianChart.viewStats(storage.getStatistics());
    
    
    
  }

}
