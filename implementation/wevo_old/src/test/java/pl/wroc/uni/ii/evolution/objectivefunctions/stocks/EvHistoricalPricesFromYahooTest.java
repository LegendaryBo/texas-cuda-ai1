package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import java.util.Date;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvHistoricalPricesFromYahoo;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvMonth;

// !!!NOTE!!! this test are turned off, since the code testes isnt working anymore. Yahoo is permuting in some way values of shares

public class EvHistoricalPricesFromYahooTest extends TestCase {
  @SuppressWarnings("deprecation")
  public void testGetPrices(){
    
    EvHistoricalPricesFromYahoo historical = new EvHistoricalPricesFromYahoo();
    historical.getPricesFromYahooFinance("MSFT", 27, EvMonth.SEP, 2006, 1, EvMonth.DEC, 2006);
    //assertEquals( 27.25f, historical.getPriceFor("MSFT", 27, EvMonth.SEP, 2006));
   
  }
  
  @SuppressWarnings("deprecation")
  public void testGetCommonSubset() {
    EvHistoricalPricesFromYahoo historical = new EvHistoricalPricesFromYahoo();
    historical.getPricesFromYahooFinance("MSFT", 2, EvMonth.NOV, 1999, 10, EvMonth.OCT, 2006);
    historical.getPricesFromYahooFinance("GOOG", 2, EvMonth.NOV, 1999, 10, EvMonth.OCT, 2006);
    historical.getCommonSubset(new Date(1999, 11 , 2),new Date(2006, 10 , 10));
    
    //assertEquals( 27.25f, historical.getPriceFor("MSFT", 27, EvMonth.SEP, 2006));
    //assertEquals(0.0f, historical.getPriceFor("MSFT", 30, EvMonth.NOV, 2001)); 
     
  }
}
