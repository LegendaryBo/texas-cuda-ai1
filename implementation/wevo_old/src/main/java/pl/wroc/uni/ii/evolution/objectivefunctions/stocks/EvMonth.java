package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

/**
 * Enum months used in HistoricalPricesFromYahoo
 * 
 * @author Marek Chrusciel, Donata Malecka
 */

public enum EvMonth {
  JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC;

  public String toString() {
    int num = this.ordinal();
    if (num < 10) {
      return "0" + num;
    }

    return "" + ordinal();
  }


  public static EvMonth valueOf(int month) {
    switch (month) {
      case 0:
        return EvMonth.JAN;
      case 1:
        return EvMonth.FEB;
      case 2:
        return EvMonth.MAR;
      case 3:
        return EvMonth.APR;
      case 4:
        return EvMonth.MAY;
      case 5:
        return EvMonth.JUN;
      case 6:
        return EvMonth.JUL;
      case 7:
        return EvMonth.AUG;
      case 8:
        return EvMonth.SEP;
      case 9:
        return EvMonth.OCT;
      case 10:
        return EvMonth.NOV;
      case 11:
        return EvMonth.DEC;
    }
    return EvMonth.JAN;
  }
}
