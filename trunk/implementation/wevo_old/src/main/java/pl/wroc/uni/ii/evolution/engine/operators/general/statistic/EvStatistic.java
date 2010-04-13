package pl.wroc.uni.ii.evolution.engine.operators.general.statistic;

import java.io.Serializable;

/**
 * Abstract class for every statistics in wEvo.
 * 
 * @author Marcin Golebiowski
 */
public abstract class EvStatistic implements Serializable {

  private int iteration;

  private long time;


  /**
   * Returns the number of iteration when statistic was generated
   * 
   * @return int
   */
  public int getIteration() {
    return iteration;
  }


  public long getTime() {
    return time;
  }


  /**
   * Sets the number of interation when statistic was generated
   * 
   * @param iteration
   */
  public void setIteration(int iteration) {
    this.iteration = iteration;
  }


  public void setTime(long time) {
    this.time = time;
  }

}
