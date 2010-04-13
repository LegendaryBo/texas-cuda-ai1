package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

/**
 * Pair class
 * 
 * @author Marcin Golebiowski
 */
public class EvPair {
  /**
   * first element of pair
   */
  public int x;

  /**
   * second element of pair
   */
  public int y;


  /**
   * constructor that sets the <code>x</code> and <code>y</code> values
   * 
   * @param x parameter to set
   * @param y parameter to set
   */
  public EvPair(int x, int y) {
    this.x = x;
    this.y = y;
  }
}
