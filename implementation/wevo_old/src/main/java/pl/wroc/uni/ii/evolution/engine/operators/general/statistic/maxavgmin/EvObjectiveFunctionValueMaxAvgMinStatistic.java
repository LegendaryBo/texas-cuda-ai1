package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * A statistic about population that contains:
 * <ul>
 * <li> max obj value
 * <li> avg obj value
 * <li> min obj value
 * <li> stdv
 * </ul>
 * 
 * @author Marcin Golebiowski
 */
public class EvObjectiveFunctionValueMaxAvgMinStatistic extends EvStatistic {

  /**
   * 
   */
  private static final long serialVersionUID = -7281904109367262854L;

  private double max, avg, min, stdev;


  public double getAvg() {
    return avg;
  }


  public double getMax() {
    return max;
  }


  public double getMin() {
    return min;
  }


  public double getStdev() {
    return stdev;
  }


  public EvObjectiveFunctionValueMaxAvgMinStatistic(int iteration, double max,
      double avg, double min, double stdev) {
    this.setIteration(iteration);
    this.min = min;
    this.max = max;
    this.avg = avg;
    this.stdev = stdev;
  }
}
