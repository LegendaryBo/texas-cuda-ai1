package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

/**
 * @author Marek Chrusciel, Donata Malecka
 */
public interface EvHistoricalPrices {
  float getPriceFor(String stock_code, int day, EvMonth month, int year);
}
