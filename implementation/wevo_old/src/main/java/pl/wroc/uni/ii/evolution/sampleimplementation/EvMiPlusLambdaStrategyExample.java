package pl.wroc.uni.ii.evolution.sampleimplementation;

/**
 * @author Lukasz Witko, Tomasz Kozakiewicz
 */
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies.EvMiPlusLambdaStrategy;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorWithProbabilitiesSpace;

public class EvMiPlusLambdaStrategyExample {

  public static void main(String[] args) {
    EvTask strategy_task = new EvTask();

    EvRealOneMax<EvRealVectorWithProbabilitiesIndividual> objective_function =
        new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>();

    EvMiPlusLambdaStrategy strategy =
        new EvMiPlusLambdaStrategy(50, 100, .2, .3);

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
