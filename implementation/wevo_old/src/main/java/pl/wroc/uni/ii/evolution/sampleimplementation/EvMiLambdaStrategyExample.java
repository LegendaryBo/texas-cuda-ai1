package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies.EvMiLambdaStrategy;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorWithProbabilitiesSpace;

/**
 * @author Lukasz Witko, Tomasz Kozakiewicz
 */
public class EvMiLambdaStrategyExample {

  public static void main(String[] args) {
    EvTask strategy_task = new EvTask();

    EvRealOneMax<EvRealVectorWithProbabilitiesIndividual> objective_function =
        new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>();

    EvMiLambdaStrategy strategy = new EvMiLambdaStrategy(50, 100, .2, .3);

    strategy
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvRealVectorWithProbabilitiesIndividual>(
            System.out));
    strategy.setSolutionSpace(new EvRealVectorWithProbabilitiesSpace(
        objective_function, 10));
    strategy.setObjectiveFunction(objective_function);
    strategy
        .setTerminationCondition(new EvMaxIteration<EvRealVectorWithProbabilitiesIndividual>(
            500));

    strategy_task.setAlgorithm(strategy);
    strategy_task.run();
    strategy_task.printBestResult();
    System.out.println(strategy_task.getBestResult()
        .getObjectiveFunctionValue());
  }
}
