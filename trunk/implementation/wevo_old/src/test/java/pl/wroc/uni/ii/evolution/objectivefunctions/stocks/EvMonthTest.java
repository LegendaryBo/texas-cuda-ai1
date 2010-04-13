package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import junit.framework.TestCase;
public class EvMonthTest extends TestCase {
  
  public void test() {

    EvMonth.valueOf("JAN");
    EvMonth.valueOf("FEB");
    EvMonth.valueOf("APR");
    EvMonth.valueOf("MAR");
    EvMonth.valueOf("MAY");
    EvMonth.valueOf("JUN");
    EvMonth.valueOf("JUL");
    EvMonth.valueOf("AUG");
    EvMonth.valueOf("SEP");
    EvMonth.valueOf("OCT");
    EvMonth.valueOf("NOV");
    EvMonth.valueOf("DEC");
    
    
    EvMonth.valueOf(0);
    EvMonth.valueOf(1);
    EvMonth.valueOf(2);
    EvMonth.valueOf(3);
    EvMonth.valueOf(4);
    EvMonth.valueOf(5);
    EvMonth.valueOf(6);
    EvMonth.valueOf(7);
    EvMonth.valueOf(8);
    EvMonth.valueOf(9);
    EvMonth.valueOf(10);
    EvMonth.valueOf(11);

  }

}
