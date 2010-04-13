package pl.wroc.uni.ii.evolution.utils;

/**
 * Interface for testing purposes.
 * 
 * @author Marek Szyku³a
 */
public interface EvIRandomizer {

  /**
   * Returns random integer value in the given interval [0, max).
   * 
   * @param max -- end of interval
   * @return random integer value
   */
  int nextInt(int max);
}
