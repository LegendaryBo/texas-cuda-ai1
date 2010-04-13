package pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies.EvOnePlusOneStrategy;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvOnePlusOneStrategyTest extends TestCase {
  
  public void testStrategy() {
    try {
      EvOnePlusOneStrategy strategy = new EvOnePlusOneStrategy(0.001, 5, 1 / 0.82, 0.82);
      strategy.setSolutionSpace(new EvRealVectorSpace(new EvRealOneMax<EvRealVectorIndividual>(), 10));
      strategy.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
      strategy.setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(100));
      
      
      EvTask task = new EvTask();
      task.setAlgorithm(strategy);
      task.run();
      task.printBestResult();
      System.out.println(task.getBestResult().getObjectiveFunctionValue());
    } catch (Exception e) {
      fail(e.getMessage());
    }
  }
}
