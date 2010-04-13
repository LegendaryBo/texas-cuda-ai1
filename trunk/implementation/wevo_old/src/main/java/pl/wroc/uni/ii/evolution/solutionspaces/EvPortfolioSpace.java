package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.ArrayList;
import java.util.Date;
import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvHistoricalPricesFromYahoo;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvMonth;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvPorfolioValue;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space useful in portfolio optimalization
 * 
 * @author Marcin Golebiowski, Marcin Brodziak
 */

public class EvPortfolioSpace implements
    EvSolutionSpace<EvNaturalNumberVectorIndividual> {

  private static final long serialVersionUID = 8772247806757736576L;

  private Double[] stock_prices;

  private double min_value = Double.MAX_VALUE;

  private String[] stock_names;

  private double start_funds;

  private Date start_investing_date;

  private Date end_investing_date;

  private EvHistoricalPricesFromYahoo prices;


  /**
   * Constructs space for individuals describing content of portfolio. At least
   * one stock must be listed on the stock exchange at start_investing_date
   * 
   * @param start_funds how much money we hava at begging
   * @param start_investing_date
   * @param end_investing_date
   * @param stock_codes list of stock's codes in our portfolio
   * @param prices_loader
   */
  @SuppressWarnings("deprecation")
  public EvPortfolioSpace(double start_funds, Date start_investing_date,
      Date end_investing_date, String[] stock_codes,
      EvHistoricalPricesFromYahoo prices_loader) {

    Date from = (Date) start_investing_date.clone();
    Date to = (Date) end_investing_date.clone();

    // fetch prices for stocks
    for (String stock_code : stock_codes) {
      prices_loader.getPricesFromYahooFinance(stock_code, start_investing_date
          .getDate(), EvMonth.valueOf(start_investing_date.getMonth()),
          start_investing_date.getYear(), end_investing_date.getDate(), EvMonth
              .valueOf(end_investing_date.getMonth()), end_investing_date
              .getYear());

    }

    ArrayList<String> codes_tmp = new ArrayList<String>();
    ArrayList<Double> list2 = new ArrayList<Double>();

    // find commonsubset
    prices_loader.getCommonSubset(from, to);

    // check if stock have price at start_investing_date
    for (String stock_code : stock_codes) {

      double value =
          prices_loader.getPriceFor(stock_code, start_investing_date.getDate(),
              EvMonth.valueOf(start_investing_date.getMonth()),
              start_investing_date.getYear());

      if (value != 0) {
        list2.add(value);
        codes_tmp.add(stock_code);
      }
    }
    if (codes_tmp.size() == 0) {
      throw new IllegalArgumentException("Sets of stocks invalid");
    }

    this.stock_prices = new Double[codes_tmp.size()];
    this.stock_names = new String[list2.size()];

    for (int i = 0; i < codes_tmp.size(); i++) {
      this.stock_names[i] = codes_tmp.get(i);
      this.stock_prices[i] = list2.get(i);
      if (this.stock_prices[i] < min_value) {
        this.min_value = this.stock_prices[i];
      }
    }

    this.prices = prices_loader;
    this.start_funds = start_funds;
    this.start_investing_date = start_investing_date;
    this.end_investing_date = end_investing_date;

  }


  public boolean belongsTo(EvNaturalNumberVectorIndividual individual) {
    double value_of_portfolio = 0;
    if (individual.getDimension() != stock_names.length) {
      return false;
    }
    for (int i = 0; i < individual.getDimension(); i++) {
      value_of_portfolio += stock_prices[i] * individual.getNumberAtPosition(i);
    }
    return value_of_portfolio <= start_funds;
  }


  public Set<EvSolutionSpace<EvNaturalNumberVectorIndividual>> divide(int n) {

    return null;
  }


  public Set<EvSolutionSpace<EvNaturalNumberVectorIndividual>> divide(int n,
      Set<EvNaturalNumberVectorIndividual> p) {

    return null;
  }


  /**
   * Generates individuals describing some portfolio. The value of this
   * portfolio doesn't exceeds the start funds.
   */
  public EvNaturalNumberVectorIndividual generateIndividual() {

    EvNaturalNumberVectorIndividual generated =
        new EvNaturalNumberVectorIndividual(stock_names.length);
    double funds =
        (double) EvRandomizer.INSTANCE.nextInt((int) this.start_funds);
    int number_of_checks = 0;
    while ((this.start_funds > min_value)
        && (number_of_checks < stock_names.length)) {
      int stock_to_buy = EvRandomizer.INSTANCE.nextInt(stock_names.length);
      int posible = (int) (funds / stock_prices[stock_to_buy]);
      if (posible == 0) {
        number_of_checks += 1;
        continue;
      }
      int volume = EvRandomizer.INSTANCE.nextInt(posible);
      funds -= volume * stock_prices[stock_to_buy];
      generated.setNumberAtPosition(stock_to_buy, volume
          + generated.getNumberAtPosition(stock_to_buy));
      number_of_checks += 1;
    }
    return generated;
  }


  /**
   * Getting back through randomly decreasing amount of shares until first day
   * has shares of value lower than the initial amount of money.
   */
  @SuppressWarnings("deprecation")
  public EvNaturalNumberVectorIndividual takeBackTo(
      EvNaturalNumberVectorIndividual individual) {
    EvPorfolioValue fun = new EvPorfolioValue(this);
    EvNaturalNumberVectorIndividual result = individual.clone();
    while (fun.portfolioValueAt(start_investing_date.getDate(), EvMonth
        .valueOf(start_investing_date.getMonth()), start_investing_date
        .getYear(), individual) < start_funds) {
      int i = EvRandomizer.INSTANCE.nextInt(individual.getDimension());
      result.setNumberAtPosition(i, result.getNumberAtPosition(i) - 1);
    }
    return result;
  }


  public String[] getStockNames() {
    return stock_names;
  }


  public double getStartFunds() {
    return start_funds;
  }


  public EvHistoricalPricesFromYahoo getPrices() {
    return prices;
  }


  public Date getEndInvestingDate() {
    return end_investing_date;
  }


  public Date getStartInvestingDate() {
    return start_investing_date;
  }


  public ArrayList<Date> getCommonDays() {
    return prices.getCommonDays();
  }


  /**
   * id doesn't work
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvNaturalNumberVectorIndividual> objective_function) {

  }


  /**
   * id doesn't work
   */
  public EvObjectiveFunction<EvNaturalNumberVectorIndividual> getObjectiveFuntion() {
    return null;
  }
}