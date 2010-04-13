package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.totaltime;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * In future it could be usefull for others. It will be tested in not too
 * distant future.
 * 
 * @author Tomasz Kozakiewicz
 */
public class EvTotalComputationTimeStatistic extends EvStatistic {

  private static final long serialVersionUID = -5435220480835060782L;

  private long node_time;


  protected EvTotalComputationTimeStatistic(long node_current_time) {
    this.node_time = node_current_time;
  }


  public long getTimeOnNode() {
    return node_time;
  }
}
