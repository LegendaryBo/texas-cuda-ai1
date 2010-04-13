package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryBMDAProbability;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Example of use EvBMDA.
 * @author Mateusz Poslednik (mateusz.poslednik@gmail.com)
 *
 */
public final class EvBMDAExample {

  /**
   * Default constructor.
   */
  private EvBMDAExample() {
    
  }
  
  /**
   * main program.
   * @param args arguments
   */
  public static void main(final String[] args) {
    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);
    
    int pop_size = 4000;
    int bits = 40;
    int iterations = 16;

    EvAlgorithm<EvBinaryVectorIndividual> alg = 
      new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);

    EvBinaryVectorSpace solution_space = 
        new EvBinaryVectorSpace(objective_function, bits);
    EvBinaryBMDAProbability bmda = 
        new EvBinaryBMDAProbability(pop_size);
    EvReplacementComposition<EvBinaryVectorIndividual> replacement = 
        new EvReplacementComposition<EvBinaryVectorIndividual>(
            bmda,
            new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());        
 
    alg.setSolutionSpace(solution_space);
    alg.setObjectiveFunction(objective_function);
    alg.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(iterations));
    alg.addOperator(replacement);
    alg.addOperator(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));

    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    bmda.collectBayesianStats(storage);
    
    
    long cur_time = System.currentTimeMillis();
    alg.init();
    alg.run();
    long after_time = System.currentTimeMillis();
    
    EvBayesianChart.viewStats(storage.getStatistics());
    
    System.out.println("RUN TIME: " 
        + ((after_time - cur_time) / 1000) + " sec");
  }

}
