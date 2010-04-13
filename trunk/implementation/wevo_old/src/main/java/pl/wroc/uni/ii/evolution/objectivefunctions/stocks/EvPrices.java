package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Date;
import java.util.Hashtable;

/**
 * @author Marek Chruœciel, Donata Ma³ecka
 */
public class EvPrices {

  /*
   * String from, to : dates - format DDMMYYYY
   */
  private String from;

  String to;

  String symbol;

  /*
   * prices container - key - date, format like from and to : DDMMYYYY
   */

  Hashtable<String, ArrayList<Number>> prices;

  /*
   * Container for dates where we don't have prices for current stock it will
   * help in choosing largest common subset for all Prices
   */
  String[] freeDates;


  public String getFrom() {
    return from;
  }


  public String getTo() {
    return to;
  }


  public String getSymbol() {
    return symbol;
  }


  /**
   * @param symbol symbol from yahoo finance
   * @param from_day from day
   * @param from_month from month (enum)
   * @param from_year from year
   * @param to_day to day
   * @param to_month to month (month enum)
   * @param to_year to year
   */
  public EvPrices(String symbol, int from_day, EvMonth from_month,
      int from_year, int to_day, EvMonth to_month, int to_year) {
    this.symbol = symbol;
    from = dayToString(from_day) + from_month.toString() + from_year;
    to = dayToString(to_day) + to_month.toString() + to_year;
    prices = new Hashtable<String, ArrayList<Number>>();
    getPrices();
  }


  public void getPrices() {
    try {
      URL yahoo;

      yahoo = new URL(getYahooFinanceURL());
      BufferedReader in =
          new BufferedReader(new InputStreamReader(yahoo.openStream()));
      String inputLine;
      // we dont want the first line
      in.readLine();
      while ((inputLine = in.readLine()) != null)
        parseFinansalYahoo(inputLine);

      in.close();
    } catch (Exception e) {
      e.printStackTrace();

    }
  }


  public String parseFinansalYahooDate(String date) {
    String[] dateTmp = date.split("-");
    String key = "";

    for (int i = dateTmp.length - 1; i >= 0; i--) {
      key += dateTmp[i];
    }
    return key;
  }


  private void parseFinansalYahoo(String line) {
    String[] tmp = line.split(",");
    String key = parseFinansalYahooDate(tmp[0]);
    long volume = Long.parseLong(tmp[5]);
    float close = Float.parseFloat(tmp[6]);
    toHashTable(key, volume, close);

  }


  private void toHashTable(String key, long volume, float close) {
    ArrayList<Number> value = new ArrayList<Number>();

    value.add(new Long(volume));
    value.add(new Float(close));

    prices.put(key, value);

  }


  public ArrayList getPrices(int day, EvMonth month, int year) {
    String date = dayToString(day) + dayToString((month.ordinal() + 1)) + year;
    ArrayList alist = prices.get(date);
    return alist;
  }


  public long getPriceVolume(int day, EvMonth month, int year) {
    String date = dayToString(day) + dayToString((month.ordinal() + 1)) + year;
    ArrayList alist = prices.get(date);
    return ((Long) alist.get(0)).longValue();
  }


  public float getPriceClose(int day, EvMonth month, int year) {
    String date = dayToString(day) + dayToString((month.ordinal() + 1)) + year;

    ArrayList alist = prices.get(date);
    if (alist == null)
      return 0.0f;

    return ((Float) alist.get(1)).floatValue();
  }


  private String dayToString(int day) {
    if (day < 10)
      return "0" + day;
    return "" + day;
  }


  private String getYahooFinanceURL() {
    StringBuffer url =
        new StringBuffer("http://ichart.finance.yahoo.com/table.csv");
    url.append("?s=" + symbol);
    url.append("&a=" + (from.substring(2, 4)).toString());
    url.append("&b=" + (from.substring(0, 2)).toString());
    url.append("&c=" + (from.substring(4, 8)).toString());

    url.append("&d=" + (to.substring(2, 4)).toString());
    url.append("&e=" + (to.substring(0, 2)).toString());
    url.append("&f=" + (to.substring(4, 8)).toString() + "&ignore=.csv");

    return url.toString();
    // ?s=MSFT&d=10&e=10&f=2006&g=d&a=2&b=13&c=1986&ignore=.csv
  }


  @SuppressWarnings("deprecation")
  private String getStringFromDate(Date d) {
    String s =
        dayToString(d.getDate()) + dayToString(d.getMonth() + 1) + d.getYear();
    return s;
  }


  /**
   * @param d Date
   * @return method check if there is a record for given date d
   */
  public boolean isInPrices(Date d) {
    String date = getStringFromDate(d);
    if (prices.get(date) == null)
      return false;
    return true;

  }


  /**
   * @param d method delete record for given day
   */
  public void deleteFromPrices(Date d) {
    prices.remove(getStringFromDate(d));
  }
}
