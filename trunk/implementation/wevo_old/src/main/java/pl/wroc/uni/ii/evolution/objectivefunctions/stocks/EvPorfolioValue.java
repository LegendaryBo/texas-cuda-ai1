package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import java.util.Date;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPortfolioSpace;

/**
 * Compute value of portfolio at given day.
 * 
 * @author Marcin Golebiowski, Marcin Brodziak
 */
public class EvPorfolioValue implements
    EvObjectiveFunction<EvNaturalNumberVectorIndividual> {

  private static final long serialVersionUID = 7164267565961436140L;

  private EvPortfolioSpace space;


  /**
   * @param space PortfolioSpace
   * @param apply_date
   */
  public EvPorfolioValue(EvPortfolioSpace space) {
    this.space = space;
  }


  /**
   * Computes average value of portfolio over a period of time considered inside
   * specific space.
   */
  @SuppressWarnings("deprecation")
  public double evaluate(EvNaturalNumberVectorIndividual individual) {
    double average_portfolio = 0;
    int counter = 0;

    for (Date d : space.getCommonDays()) {

      average_portfolio +=
          portfolioValueAt(d.getDate(), EvMonth.valueOf(d.getMonth()), d
              .getYear(), individual);
      counter++;
    }
    average_portfolio /= counter;
    return average_portfolio;
  }


  /**
   * ??
   * 
   * @param d
   * @param individual
   * @return
   */
  public double portfolioValueAt(int day, EvMonth month, int year,
      EvNaturalNumberVectorIndividual individual) {
    EvHistoricalPricesFromYahoo historical = space.getPrices();
    String[] stock_names = space.getStockNames();
    double value_of_portfolio_at_end = 0;
    for (int i = 0; i < individual.getDimension(); i++) {
      value_of_portfolio_at_end +=
          historical.getPriceFor(stock_names[i], day, month, year)
              * individual.getNumberAtPosition(i);
    }
    return value_of_portfolio_at_end;
  }
}
