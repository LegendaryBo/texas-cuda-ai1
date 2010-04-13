package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import java.util.ArrayList;
import java.util.Date;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Set;

/**
 * @author Marek Chrusciel, Donata Malecka Class is used to get stock prices
 *         from yahoo finance
 */
public class EvHistoricalPricesFromYahoo implements EvHistoricalPrices {

  private Hashtable<String, EvPrices> stock_prices;

  private ArrayList<Date> common_days = new ArrayList<Date>();


  /**
   * basic constructor
   */
  public EvHistoricalPricesFromYahoo() {
    stock_prices = new Hashtable<String, EvPrices>();
  }


  /**
   * return adj close from downloaded data. if there is no data for the stock it
   * returns 0
   */
  @SuppressWarnings("deprecation")
  public float getPriceFor(String stock_code, int day, EvMonth month, int year) {
    if (stock_prices.get(stock_code) == null) {
      return 0.0f;
    }
    return stock_prices.get(stock_code).getPriceClose(day, month, year);
  }


  @SuppressWarnings("deprecation")
  public long getVolumeFor(String stock_code, Date date) {
    return stock_prices.get(stock_code).getPriceVolume(date.getDate(),
        EvMonth.valueOf(date.getMonth()), date.getYear());
  }


  /**
   * get Adj. Close and volume value from yahoo finance for given stock code and
   * given dates
   */
  @SuppressWarnings("deprecation")
  public void getPricesFromYahooFinance(String symbol, int d1, EvMonth m1,
      int year1, int d2, EvMonth m2, int year2) {
    if (stock_prices.get(symbol) != null) {
      stock_prices.remove(symbol);
    }

    EvPrices prices = new EvPrices(symbol, d1, m1, year1, d2, m2, year2);
    stock_prices.put(symbol, prices);
  }


  /**
   * @param from
   * @param to this method delete all record in stock prices for a given date
   *        where there is at least one stock with no record for this day
   *        (largest common subset)
   */
  public void getCommonSubset(Date from, Date to) {
    Set<String> keys = stock_prices.keySet();
    Date tmp = from;
    common_days = new ArrayList<Date>();
    while (!tmp.after(to)) {
      Iterator<String> ikeys = keys.iterator();

      boolean deleted = false;
      while (ikeys.hasNext()) {
        String key = ikeys.next();
        if (!stock_prices.get(key).isInPrices(tmp)) {
          // delete the date prices from all stocks
          deleted = true;
          Iterator<String> ikeys2 = keys.iterator();
          while (ikeys2.hasNext())
            stock_prices.get(ikeys2.next()).deleteFromPrices(tmp);
          break;
        }
      }
      if (!deleted) {
        common_days.add((Date) tmp.clone());
      }
      // increment the tmp Date by one day
      long time = tmp.getTime() + 1000 * 60 * 60 * 24;
      tmp.setTime(time);
    }
  }


  public ArrayList<Date> getCommonDays() {
    return this.common_days;
  }

}
