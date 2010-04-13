package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies.EvMiLambdaRoKappaStrategy;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvMiLambdaRoKappaSpace;

/**
 * @author Lukasz Witko
 */
public class EvMiLambdaRoKappaStrategyExample {

  public static void main(String[] args) {
    EvTask strategy_task = new EvTask();

    EvMiLambdaRoKappaStrategy strategy =
        new EvMiLambdaRoKappaStrategy(50, 100, 5, .25, .16, 0.0873, 5, false);

    strategy
        .setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    strategy.setSolutionSpace(new EvMiLambdaRoKappaSpace(
        new EvRealOneMax<EvMiLambdaRoKappaIndividual>(), 10));

    strategy
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvMiLambdaRoKappaIndividual>(
            System.out));
    strategy
        .setTerminationCondition(new EvMaxIteration<EvMiLambdaRoKappaIndividual>(
            50));

    strategy_task.setAlgorithm(strategy);
    strategy_task.run();
    strategy_task.printBestResult();
    System.out.println(strategy_task.getBestResult()
        .getObjectiveFunctionValue());
  }
}
