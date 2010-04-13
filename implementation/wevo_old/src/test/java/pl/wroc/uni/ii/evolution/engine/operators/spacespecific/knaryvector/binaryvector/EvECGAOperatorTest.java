package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorECGAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import junit.framework.TestCase;

/**
 * ECGA operator tests.
 * 
 * @author Kacper Gorski
 */
public class EvECGAOperatorTest extends TestCase {

  public void testECGA() {
    final int pop_size = 2000;
    final int bits = 32;

    EvAlgorithm<EvBinaryVectorIndividual> alg = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    EvKDeceptiveOneMax one_max = new EvKDeceptiveOneMax(4);
    EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
    EvBinaryVectorECGAOperator ecga = new EvBinaryVectorECGAOperator(16, false, solution_space);
    alg.setObjectiveFunction(one_max);
    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(System.out));
    alg.addOperatorToEnd(ecga);
    
    alg.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(10));
    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(alg);
    evolutionary_task.run();
    
    assertEquals("00000000000000000000000000000000", alg.getBestResult().toString());
    
  }

}
