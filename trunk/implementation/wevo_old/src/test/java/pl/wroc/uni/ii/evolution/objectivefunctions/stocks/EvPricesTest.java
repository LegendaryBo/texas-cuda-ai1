package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvMonth;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvPrices;

public class EvPricesTest extends TestCase {
  
  EvPrices test;
  
  @Override
  protected void setUp(){
    test = new EvPrices("MSFT", 13, EvMonth.JAN, 1990, 2, EvMonth.OCT, 2005);
  }
  
  public void testConstructior(){
    String expectedFromDate  = "13001990";
    String expectedToDate = "02092005";
    String expectedSymbol = "MSFT";
    assertEquals("wrong date from",expectedFromDate, test.getFrom());
    assertEquals("wrong date to",expectedToDate, test.getTo());
    assertEquals("wrong symbol",expectedSymbol, test.getSymbol());
  }
  
  public void testGetPrices(){
    test.getPrices();
    
  }
  
  public void testParseFinansalYahooDate(){
    String date1 = test.parseFinansalYahooDate("2005-01-05");
    String date2 = test.parseFinansalYahooDate("2005-07-22");
    String date3 = test.parseFinansalYahooDate("1999-07-22");
    
    assertEquals(date1, "05012005");
    assertEquals(date2, "22072005");
    assertEquals(date3, "22071999");
    
  }
  
  

}
