package pl.wroc.uni.ii.evolution.sampleimplementation.comparision;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Sample of ECGA algorithm used in comparision to Illegal's implementation.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 *
 */
public class EvECGAComparisionExample {
  
  /**
   * Disabling constructor.
   */
  protected EvECGAComparisionExample() {
    throw new IllegalStateException("Class shouldnt be instantiated");
  }
  
  /**
   * runs algorithm and display some statistics.
   * 
   * @param args - not used
   */
  public static void main(final String[] args) {

    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);

    EvAlgorithm<EvBinaryVectorIndividual> ecga = new EvECGA(false, 4000, 16, 1);

    ecga.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 60));
    ecga.setObjectiveFunction(objective_function);
    ecga.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(7));
    
    ecga.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    
    long cur_time = System.currentTimeMillis();
    ecga.init();
    ecga.run();
    long after_time = System.currentTimeMillis();
    
    System.out.println("RUN TIME: " + (after_time - cur_time) + " msec");
  }
}
