package pl.wroc.uni.ii.evolution.sampleimplementation;

import java.util.Date;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.misc.EvTakeBackToSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvTournamentSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorChangeMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvHistoricalPricesFromYahoo;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvMonth;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvPorfolioValue;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPortfolioSpace;

/**
 * Portfolio optimization example
 */

public class EvPortfolioOptimalization {

  /**
   * @param args
   */

  @SuppressWarnings("all")
  public static void main(String[] args) {
    EvTask evolutionary_task = new EvTask();

    EvAlgorithm<EvNaturalNumberVectorIndividual> genericEA =
        new EvAlgorithm<EvNaturalNumberVectorIndividual>(40);

    System.out.println("Init space: Start");
    EvPortfolioSpace space =
        new EvPortfolioSpace(100000, new Date(2005, 8, 1),
            new Date(2006, 9, 10), new String[] {"MSFT", "GOOG", "XOM", "SWY"},
            new EvHistoricalPricesFromYahoo());
    System.out.println("Init space: Done = " + space.getStockNames().length);
    genericEA.setSolutionSpace(space);

    EvPorfolioValue fun = new EvPorfolioValue(space);

    genericEA.setObjectiveFunction(fun);
    genericEA
        .addOperatorToEnd(new EvTournamentSelection<EvNaturalNumberVectorIndividual>(
            4, 1));
    genericEA.addOperatorToEnd(new EvNaturalNumberVectorUniformCrossover());
    genericEA.addOperatorToEnd(new EvNaturalNumberVectorChangeMutation(0.4, 5));

    genericEA
        .addOperatorToEnd(new EvTakeBackToSolutionSpace<EvNaturalNumberVectorIndividual>(
            (EvSolutionSpace<EvNaturalNumberVectorIndividual>) space));

    genericEA
        .setTerminationCondition(new EvMaxIteration<EvNaturalNumberVectorIndividual>(
            10));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();
    System.out.println(fun.portfolioValueAt(1, EvMonth.OCT, 2005,
        (EvNaturalNumberVectorIndividual) genericEA.getBestResult()));
    EvPorfolioValue fun_start = new EvPorfolioValue(space);
    System.out.println(fun_start
        .evaluate((EvNaturalNumberVectorIndividual) genericEA.getBestResult()));
  }

}
