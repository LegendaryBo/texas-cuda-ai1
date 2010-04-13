package pl.wroc.uni.ii.evolution.experimental;

import java.util.Date;

/**
 * Represents a history of playing on a stock market, that is buying and selling
 * shares. It has a sum of money at start. For each transaction there is time,
 * number of stocks, and company name stored. The end date is stored also, in
 * order to know until what date to evaluate the performance.
 * 
 * @author Kamil Dworakowski
 */
public class EvInvestmentsHistoryIndividual {

  Date end_date;

  float start_funds;


  /**
   * @param start_funds how much money at start
   * @param end_date the performance will be evaluated to that date
   */
  public EvInvestmentsHistoryIndividual(float start_funds, Date end_date) {
    this.end_date = end_date;
    this.start_funds = start_funds;
  }


  public float getAssetsValueAtEnd() {
    return start_funds;
  }


  /**
   * Says that the event took place, namely that given number of shares has been
   * bought on the given day. The code refers to the accronym used on the stock
   * market for the stock.
   * 
   * @param stock_code code for the stock bought
   * @param date of transaction
   * @param number of shares bought
   */
  public void bought_shares(String stock_code, Date date, int number) {
  }


  /**
   * Put the sell event on record.
   * 
   * @param stock_code code for the stock sold
   * @param date of transaction
   * @param number of shares sold
   */
  public void sold_shares(String stock_code, Date date, int number) {
  }
}
