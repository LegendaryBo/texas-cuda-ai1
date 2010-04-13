package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import pl.wroc.uni.ii.evolution.experimental.EvInvestmentsHistoryIndividual;

/**
 * A function that evaluates QuoteIndividuals. It returns the value of the
 * assets at the end of a investment history.
 * 
 * @author Kamil Dworakowski
 */
public class EvInvestmentsObjectiveFunction {

  EvHistoricalPrices historical_prices;


  /**
   * @param prices the date from which to stora historical prices
   */
  public EvInvestmentsObjectiveFunction(EvHistoricalPrices prices) {
    this.historical_prices = prices;
  }


  public double evaluate(EvInvestmentsHistoryIndividual individual) {
    return individual.getAssetsValueAtEnd();
  }

}
